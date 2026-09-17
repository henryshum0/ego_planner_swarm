#include <plan_env/rolling_log_odds_grid.h>

#include <plan_env/raycast.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace plan_env
{
namespace
{
double logit(const double probability)
{
  return std::log(probability / (1.0 - probability));
}

bool finiteVector(const Eigen::Vector3d & value)
{
  return std::isfinite(value.x()) && std::isfinite(value.y()) && std::isfinite(value.z());
}
}  // namespace

RollingLogOddsGrid::RollingLogOddsGrid(const RollingLogOddsGridConfig & config)
: config_(config)
{
  if (config_.resolution <= 0.0 || config_.size.x() <= 0.0 || config_.size.y() <= 0.0 ||
    config_.size.z() <= 0.0 || config_.recenter_distance <= 0.0 ||
    config_.vertical_recenter_distance <= 0.0 || config_.obstacle_ttl_sec <= 0.0)
  {
    throw std::invalid_argument(
            "rolling log-odds grid dimensions, resolution, recenter distances, and TTL must be positive");
  }

  resolution_inv_ = 1.0 / config_.resolution;
  hit_log_odds_ = logit(config_.hit_probability);
  miss_log_odds_ = logit(config_.miss_probability);
  clamp_min_log_odds_ = logit(config_.minimum_probability);
  clamp_max_log_odds_ = logit(config_.maximum_probability);
  occupied_log_odds_ = logit(config_.occupied_probability);
  // An unobserved cell has a neutral prior.  It remains traversable because
  // only cells above occupied_log_odds_ block the planner, while two direct
  // cloud hits using the configured 0.70 profile are sufficient to cross the
  // 0.80 occupied threshold.
  unknown_log_odds_ = 0.0;
  obstacle_ttl_ns_ = static_cast<std::int64_t>(config_.obstacle_ttl_sec * 1.0e9);
  horizontal_inflation_steps_ =
    static_cast<int>(std::ceil(config_.obstacle_inflation * resolution_inv_));

  voxel_count_ = Eigen::Vector3i(
    static_cast<int>(std::ceil(config_.size.x() * resolution_inv_)),
    static_cast<int>(std::ceil(config_.size.y() * resolution_inv_)),
    static_cast<int>(std::ceil(config_.size.z() * resolution_inv_)));
  origin_ =
    Eigen::Vector3d(-config_.size.x() * 0.5, -config_.size.y() * 0.5, config_.ground_height);
  map_max_ = origin_ + config_.size;
  resetStorage();
}

void RollingLogOddsGrid::resetStorage()
{
  const std::size_t count = static_cast<std::size_t>(voxel_count_.x()) * voxel_count_.y() *
    voxel_count_.z();
  raw_log_odds_.assign(count, unknown_log_odds_);
  last_hit_ns_.assign(count, 0);
  inflation_refcount_.assign(count, 0);
  virtual_ceiling_mask_.assign(count, 0);
  active_slot_.assign(count, -1);
  inflated_slot_.assign(count, -1);
  observation_stamp_.assign(count, 0);
  observation_is_hit_.assign(count, 0);
  active_occupied_.clear();
  active_inflated_.clear();
  observed_this_cloud_.clear();
  observation_generation_ = 0;
  rebuildVirtualCeiling();
}

bool RollingLogOddsGrid::recenterIfNeeded(const Eigen::Vector3d & vehicle_position)
{
  if (!finiteVector(vehicle_position)) {
    return false;
  }
  const Eigen::Vector3d center(
    origin_.x() + config_.size.x() * 0.5,
    origin_.y() + config_.size.y() * 0.5,
    origin_.z() + config_.size.z() * 0.5);
  if (std::abs(vehicle_position.x() - center.x()) < config_.recenter_distance &&
    std::abs(vehicle_position.y() - center.y()) < config_.recenter_distance &&
    std::abs(vehicle_position.z() - center.z()) < config_.vertical_recenter_distance)
  {
    return false;
  }
  recenterTo(vehicle_position);
  return true;
}

void RollingLogOddsGrid::recenterTo(const Eigen::Vector3d & vehicle_position)
{
  const Eigen::Vector3d previous_origin = origin_;
  const auto previous_raw = raw_log_odds_;
  const auto previous_hits = last_hit_ns_;
  const Eigen::Vector3d requested_center(
    std::round(vehicle_position.x() * resolution_inv_) * config_.resolution,
    std::round(vehicle_position.y() * resolution_inv_) * config_.resolution,
    std::round(vehicle_position.z() * resolution_inv_) * config_.resolution);
  origin_ = Eigen::Vector3d(
    requested_center.x() - config_.size.x() * 0.5,
    requested_center.y() - config_.size.y() * 0.5,
    requested_center.z() - config_.size.z() * 0.5);
  map_max_ = origin_ + config_.size;

  const std::size_t count = previous_raw.size();
  raw_log_odds_.assign(count, unknown_log_odds_);
  last_hit_ns_.assign(count, 0);
  inflation_refcount_.assign(count, 0);
  virtual_ceiling_mask_.assign(count, 0);
  active_slot_.assign(count, -1);
  inflated_slot_.assign(count, -1);
  observation_stamp_.assign(count, 0);
  observation_is_hit_.assign(count, 0);
  active_occupied_.clear();
  active_inflated_.clear();
  observed_this_cloud_.clear();
  observation_generation_ = 0;

  for (int x = 0; x < voxel_count_.x(); ++x) {
    for (int y = 0; y < voxel_count_.y(); ++y) {
      for (int z = 0; z < voxel_count_.z(); ++z) {
        const Eigen::Vector3i previous_index(x, y, z);
        const std::size_t previous_address =
          static_cast<std::size_t>(x) * voxel_count_.y() * voxel_count_.z() +
          static_cast<std::size_t>(y) * voxel_count_.z() + z;
        const Eigen::Vector3d world_position =
          previous_origin + (previous_index.cast<double>() + Eigen::Vector3d::Constant(0.5)) *
          config_.resolution;
        if (!isInMap(world_position)) {
          continue;
        }
        const Eigen::Vector3i new_index = positionToIndex(world_position);
        const std::size_t new_address = address(new_index);
        raw_log_odds_[new_address] = previous_raw[previous_address];
        last_hit_ns_[new_address] = previous_hits[previous_address];
      }
    }
  }
  rebuildDerivedState();
}

void RollingLogOddsGrid::integrateCloud(
  const Eigen::Vector3d & sensor_origin,
  const std::vector<Eigen::Vector3d> & points,
  const Eigen::Vector3d & observation_range,
  const std::int64_t now_ns)
{
  if (!finiteVector(sensor_origin)) {
    return;
  }
  recenterIfNeeded(sensor_origin);
  expire(now_ns);
  if (!isInMap(sensor_origin)) {
    return;
  }

  ++observation_generation_;
  if (observation_generation_ == 0) {
    std::fill(observation_stamp_.begin(), observation_stamp_.end(), 0);
    observation_generation_ = 1;
  }
  observed_this_cloud_.clear();

  const Eigen::Vector3d sensor_grid = (sensor_origin - origin_) * resolution_inv_;
  RayCaster raycaster;
  Eigen::Vector3d ray_point;
  for (const auto & point : points) {
    if (!finiteVector(point) || !isInMap(point)) {
      continue;
    }
    const Eigen::Vector3d deviation = point - sensor_origin;
    if (std::abs(deviation.x()) >= observation_range.x() ||
      std::abs(deviation.y()) >= observation_range.y() ||
      std::abs(deviation.z()) >= observation_range.z())
    {
      continue;
    }

    const Eigen::Vector3i endpoint_index = positionToIndex(point);
    const std::size_t endpoint_address = address(endpoint_index);
    recordObservation(endpoint_address, true);

    const Eigen::Vector3d endpoint_grid = (point - origin_) * resolution_inv_;
    if (!raycaster.setInput(endpoint_grid, sensor_grid)) {
      continue;
    }
    while (raycaster.step(ray_point)) {
      const Eigen::Vector3i ray_index = ray_point.cast<int>();
      if (!indexInBounds(ray_index) || ray_index == endpoint_index) {
        continue;
      }
      recordObservation(address(ray_index), false);
    }
  }

  for (const int observed_address : observed_this_cloud_) {
    applyObservation(
      static_cast<std::size_t>(observed_address),
      observation_is_hit_[static_cast<std::size_t>(observed_address)] != 0,
      now_ns);
  }
}

void RollingLogOddsGrid::expire(const std::int64_t now_ns)
{
  if (now_ns <= 0 || obstacle_ttl_ns_ <= 0) {
    return;
  }
  for (std::size_t i = 0; i < active_occupied_.size(); ) {
    const std::size_t voxel_address = static_cast<std::size_t>(active_occupied_[i]);
    if (last_hit_ns_[voxel_address] > 0 &&
      now_ns - last_hit_ns_[voxel_address] > obstacle_ttl_ns_)
    {
      raw_log_odds_[voxel_address] = unknown_log_odds_;
      last_hit_ns_[voxel_address] = 0;
      setRawOccupied(voxel_address, false);
      continue;
    }
    ++i;
  }
}

int RollingLogOddsGrid::getOccupancy(const Eigen::Vector3d & position) const
{
  if (!isInMap(position)) {
    return -1;
  }
  return rawOccupied(address(positionToIndex(position))) ? 1 : 0;
}

int RollingLogOddsGrid::getInflatedOccupancy(const Eigen::Vector3d & position) const
{
  if (!isInMap(position)) {
    return -1;
  }
  return inflated(address(positionToIndex(position))) ? 1 : 0;
}

bool RollingLogOddsGrid::isInMap(const Eigen::Vector3d & position) const
{
  return position.x() >= origin_.x() + 1e-4 && position.y() >= origin_.y() + 1e-4 &&
         position.z() >= origin_.z() + 1e-4 && position.x() <= map_max_.x() - 1e-4 &&
         position.y() <= map_max_.y() - 1e-4 && position.z() <= map_max_.z() - 1e-4;
}

bool RollingLogOddsGrid::rawOccupiedAt(const std::size_t voxel_address) const
{
  return voxel_address < raw_log_odds_.size() && rawOccupied(voxel_address);
}

bool RollingLogOddsGrid::inflatedAt(const std::size_t voxel_address) const
{
  return voxel_address < raw_log_odds_.size() && inflated(voxel_address);
}

Eigen::Vector3d RollingLogOddsGrid::indexToPosition(const Eigen::Vector3i & index) const
{
  return origin_ + (index.cast<double>() + Eigen::Vector3d::Constant(0.5)) * config_.resolution;
}

Eigen::Vector3d RollingLogOddsGrid::addressToPosition(const std::size_t voxel_address) const
{
  if (voxel_address >= raw_log_odds_.size()) {
    throw std::out_of_range("rolling-grid voxel address is outside the map");
  }
  return indexToPosition(indexFromAddress(voxel_address));
}

void RollingLogOddsGrid::rebuildDerivedState()
{
  clearActiveOccupied();
  rebuildVirtualCeiling();
  rebuildInflation();
}

void RollingLogOddsGrid::rebuildVirtualCeiling()
{
  std::fill(virtual_ceiling_mask_.begin(), virtual_ceiling_mask_.end(), 0);
  if (config_.virtual_ceiling_height < 0.0) {
    return;
  }
  const int ceiling_index = static_cast<int>(std::floor(
      (config_.virtual_ceiling_height - origin_.z()) * resolution_inv_)) - 1;
  if (ceiling_index < 0 || ceiling_index >= voxel_count_.z()) {
    return;
  }
  for (int x = 0; x < voxel_count_.x(); ++x) {
    for (int y = 0; y < voxel_count_.y(); ++y) {
      virtual_ceiling_mask_[address(Eigen::Vector3i(x, y, ceiling_index))] = 1;
    }
  }
}

void RollingLogOddsGrid::rebuildInflation()
{
  std::fill(inflation_refcount_.begin(), inflation_refcount_.end(), 0);
  std::fill(inflated_slot_.begin(), inflated_slot_.end(), -1);
  active_inflated_.clear();
  for (std::size_t voxel_address = 0; voxel_address < raw_log_odds_.size(); ++voxel_address) {
    if (rawOccupied(voxel_address)) {
      adjustInflation(indexFromAddress(voxel_address), 1);
    }
  }
}

void RollingLogOddsGrid::clearActiveOccupied()
{
  active_occupied_.clear();
  std::fill(active_slot_.begin(), active_slot_.end(), -1);
  for (std::size_t voxel_address = 0; voxel_address < raw_log_odds_.size(); ++voxel_address) {
    if (rawOccupied(voxel_address)) {
      addActive(voxel_address);
    }
  }
}

bool RollingLogOddsGrid::indexInBounds(const Eigen::Vector3i & index) const
{
  return index.x() >= 0 && index.y() >= 0 && index.z() >= 0 && index.x() < voxel_count_.x() &&
         index.y() < voxel_count_.y() && index.z() < voxel_count_.z();
}

Eigen::Vector3i RollingLogOddsGrid::positionToIndex(const Eigen::Vector3d & position) const
{
  return ((position - origin_) * resolution_inv_).array().floor().cast<int>();
}

std::size_t RollingLogOddsGrid::address(const Eigen::Vector3i & index) const
{
  return static_cast<std::size_t>(index.x()) * voxel_count_.y() * voxel_count_.z() +
         static_cast<std::size_t>(index.y()) * voxel_count_.z() + index.z();
}

Eigen::Vector3i RollingLogOddsGrid::indexFromAddress(const std::size_t voxel_address) const
{
  const std::size_t yz_size = static_cast<std::size_t>(voxel_count_.y()) * voxel_count_.z();
  const int x = static_cast<int>(voxel_address / yz_size);
  const std::size_t remainder = voxel_address % yz_size;
  return Eigen::Vector3i(
    x, static_cast<int>(remainder / voxel_count_.z()),
    static_cast<int>(remainder % voxel_count_.z()));
}

bool RollingLogOddsGrid::rawOccupied(const std::size_t voxel_address) const
{
  return raw_log_odds_[voxel_address] > occupied_log_odds_;
}

bool RollingLogOddsGrid::inflated(const std::size_t voxel_address) const
{
  return inflation_refcount_[voxel_address] > 0 || virtual_ceiling_mask_[voxel_address] != 0;
}

void RollingLogOddsGrid::recordObservation(const std::size_t voxel_address, const bool hit)
{
  if (observation_stamp_[voxel_address] != observation_generation_) {
    observation_stamp_[voxel_address] = observation_generation_;
    observation_is_hit_[voxel_address] = hit ? 1 : 0;
    observed_this_cloud_.push_back(static_cast<int>(voxel_address));
  } else if (hit) {
    observation_is_hit_[voxel_address] = 1;
  }
}

void RollingLogOddsGrid::applyObservation(
  const std::size_t voxel_address, const bool hit,
  const std::int64_t now_ns)
{
  const bool was_occupied = rawOccupied(voxel_address);
  const double update = hit ? hit_log_odds_ : miss_log_odds_;
  raw_log_odds_[voxel_address] = std::clamp(
    raw_log_odds_[voxel_address] + update,
    clamp_min_log_odds_, clamp_max_log_odds_);
  if (hit) {
    last_hit_ns_[voxel_address] = now_ns;
  }
  const bool is_occupied = rawOccupied(voxel_address);
  if (was_occupied != is_occupied) {
    setRawOccupied(voxel_address, is_occupied);
  }
}

void RollingLogOddsGrid::setRawOccupied(const std::size_t voxel_address, const bool occupied)
{
  const Eigen::Vector3i index = indexFromAddress(voxel_address);
  adjustInflation(index, occupied ? 1 : -1);
  if (occupied) {
    addActive(voxel_address);
  } else {
    removeActive(voxel_address);
  }
}

void RollingLogOddsGrid::addActive(const std::size_t voxel_address)
{
  if (active_slot_[voxel_address] >= 0) {
    return;
  }
  active_slot_[voxel_address] = static_cast<int>(active_occupied_.size());
  active_occupied_.push_back(static_cast<int>(voxel_address));
}

void RollingLogOddsGrid::removeActive(const std::size_t voxel_address)
{
  const int slot = active_slot_[voxel_address];
  if (slot < 0) {
    return;
  }
  const int last_address = active_occupied_.back();
  active_occupied_[static_cast<std::size_t>(slot)] = last_address;
  active_slot_[static_cast<std::size_t>(last_address)] = slot;
  active_occupied_.pop_back();
  active_slot_[voxel_address] = -1;
}

void RollingLogOddsGrid::addActiveInflated(const std::size_t voxel_address)
{
  if (inflated_slot_[voxel_address] >= 0) {
    return;
  }
  inflated_slot_[voxel_address] = static_cast<int>(active_inflated_.size());
  active_inflated_.push_back(static_cast<int>(voxel_address));
}

void RollingLogOddsGrid::removeActiveInflated(const std::size_t voxel_address)
{
  const int slot = inflated_slot_[voxel_address];
  if (slot < 0) {
    return;
  }
  const int last_address = active_inflated_.back();
  active_inflated_[static_cast<std::size_t>(slot)] = last_address;
  inflated_slot_[static_cast<std::size_t>(last_address)] = slot;
  active_inflated_.pop_back();
  inflated_slot_[voxel_address] = -1;
}

void RollingLogOddsGrid::adjustInflation(const Eigen::Vector3i & center, const int delta)
{
  for (int x = -horizontal_inflation_steps_; x <= horizontal_inflation_steps_; ++x) {
    for (int y = -horizontal_inflation_steps_; y <= horizontal_inflation_steps_; ++y) {
      for (int z = -1; z <= 1; ++z) {
        const Eigen::Vector3i index = center + Eigen::Vector3i(x, y, z);
        if (!indexInBounds(index)) {
          continue;
        }
        auto & count = inflation_refcount_[address(index)];
        if (delta > 0) {
          if (count == 0) {
            addActiveInflated(address(index));
          }
          ++count;
        } else if (count > 0) {
          --count;
          if (count == 0) {
            removeActiveInflated(address(index));
          }
        }
      }
    }
  }
}

}  // namespace plan_env
