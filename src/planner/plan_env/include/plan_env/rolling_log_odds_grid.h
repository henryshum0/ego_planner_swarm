#ifndef PLAN_ENV_ROLLING_LOG_ODDS_GRID_H_
#define PLAN_ENV_ROLLING_LOG_ODDS_GRID_H_

#include <Eigen/Eigen>

#include <cstdint>
#include <vector>

namespace plan_env
{

  struct RollingLogOddsGridConfig
  {
    Eigen::Vector3d size {Eigen::Vector3d::Zero()};
    double resolution {0.0};
    double ground_height {0.0};
    double virtual_ceiling_height {-1.0};
    double obstacle_inflation {0.0};
    double hit_probability {0.70};
    double miss_probability {0.35};
    double minimum_probability {0.12};
    double maximum_probability {0.97};
    double occupied_probability {0.80};
    double recenter_distance {10.0};
    double vertical_recenter_distance {5.0};
    double obstacle_ttl_sec {30.0};
  };

// Dense, world-aligned map whose horizontal bounds follow the vehicle.  It is
// deliberately independent of ROS messages so the map logic can be tested
// without an executor or a running planner node.
  class RollingLogOddsGrid
  {
public:
    explicit RollingLogOddsGrid(const RollingLogOddsGridConfig & config);

    bool recenterIfNeeded(const Eigen::Vector3d & vehicle_position);
    void integrateCloud(
      const Eigen::Vector3d & sensor_origin,
      const std::vector < Eigen::Vector3d > & points,
      const Eigen::Vector3d & observation_range,
      std::int64_t now_ns);
    void expire(std::int64_t now_ns);

    int getOccupancy(const Eigen::Vector3d & position) const;
    int getInflatedOccupancy(const Eigen::Vector3d & position) const;
    bool isInMap(const Eigen::Vector3d & position) const;

    const Eigen::Vector3d & origin() const {return origin_;}
    const Eigen::Vector3d & size() const {return config_.size;}
    const Eigen::Vector3i & voxelCount() const {return voxel_count_;}
    double resolution() const {return config_.resolution;}
    std::size_t voxelCountTotal() const {return raw_log_odds_.size();}
    bool rawOccupiedAt(std::size_t address) const;
    bool inflatedAt(std::size_t address) const;
    Eigen::Vector3d indexToPosition(const Eigen::Vector3i & index) const;

private:
    RollingLogOddsGridConfig config_;
    Eigen::Vector3d origin_;
    Eigen::Vector3d map_max_;
    Eigen::Vector3i voxel_count_;
    double resolution_inv_ {0.0};
    double hit_log_odds_ {0.0};
    double miss_log_odds_ {0.0};
    double clamp_min_log_odds_ {0.0};
    double clamp_max_log_odds_ {0.0};
    double occupied_log_odds_ {0.0};
    double unknown_log_odds_ {0.0};
    std::int64_t obstacle_ttl_ns_ {0};
    int horizontal_inflation_steps_ {0};

    std::vector < double > raw_log_odds_;
    std::vector < std::int64_t > last_hit_ns_;
    std::vector < std::uint16_t > inflation_refcount_;
    std::vector < std::uint8_t > virtual_ceiling_mask_;
    std::vector < int > active_occupied_;
    std::vector < int > active_slot_;

    std::vector < std::uint32_t > observation_stamp_;
    std::vector < std::uint8_t > observation_is_hit_;
    std::vector < int > observed_this_cloud_;
    std::uint32_t observation_generation_ {0};

    void resetStorage();
    void recenterTo(const Eigen::Vector3d & vehicle_position);
    void rebuildDerivedState();
    void rebuildVirtualCeiling();
    void rebuildInflation();
    void clearActiveOccupied();

    bool indexInBounds(const Eigen::Vector3i & index) const;
    Eigen::Vector3i positionToIndex(const Eigen::Vector3d & position) const;
    std::size_t address(const Eigen::Vector3i & index) const;
    Eigen::Vector3i indexFromAddress(std::size_t address) const;
    bool rawOccupied(std::size_t address) const;
    bool inflated(std::size_t address) const;

    void recordObservation(std::size_t address, bool hit);
    void applyObservation(std::size_t address, bool hit, std::int64_t now_ns);
    void setRawOccupied(std::size_t address, bool occupied);
    void addActive(std::size_t address);
    void removeActive(std::size_t address);
    void adjustInflation(const Eigen::Vector3i & center, int delta);
  };

}  // namespace plan_env

#endif  // PLAN_ENV_ROLLING_LOG_ODDS_GRID_H_
