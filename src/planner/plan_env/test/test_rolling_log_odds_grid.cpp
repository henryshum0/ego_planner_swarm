#include <gtest/gtest.h>

#include <plan_env/rolling_log_odds_grid.h>

namespace
{
using plan_env::RollingLogOddsGrid;
using plan_env::RollingLogOddsGridConfig;

constexpr std::int64_t kSecond = 1000000000LL;

RollingLogOddsGridConfig testConfig()
{
  RollingLogOddsGridConfig config;
  config.size = Eigen::Vector3d(10.0, 10.0, 4.0);
  config.resolution = 1.0;
  config.ground_height = 0.0;
  config.virtual_ceiling_height = -1.0;
  config.obstacle_inflation = 1.0;
  config.recenter_distance = 3.0;
  config.vertical_recenter_distance = 1.0;
  config.obstacle_ttl_sec = 30.0;
  return config;
}

void observeHitTwice(
  RollingLogOddsGrid & grid, const Eigen::Vector3d & sensor,
  const Eigen::Vector3d & point)
{
  const Eigen::Vector3d range(10.0, 10.0, 10.0);
  grid.integrateCloud(sensor, {point}, range, kSecond);
  grid.integrateCloud(sensor, {point}, range, 2 * kSecond);
}

TEST(RollingLogOddsGrid, HitCrossesThresholdAndInflates)
{
  RollingLogOddsGrid grid(testConfig());
  const Eigen::Vector3d obstacle(2.5, 0.5, 1.5);
  observeHitTwice(grid, Eigen::Vector3d(0.5, 0.5, 1.5), obstacle);

  EXPECT_EQ(grid.getOccupancy(obstacle), 1);
  EXPECT_EQ(grid.getInflatedOccupancy(obstacle), 1);
  EXPECT_EQ(grid.getInflatedOccupancy(Eigen::Vector3d(3.5, 1.5, 2.5)), 1);
}

TEST(RollingLogOddsGrid, FreeRayRemovesObstacleAndInflation)
{
  RollingLogOddsGrid grid(testConfig());
  const Eigen::Vector3d sensor(0.5, 0.5, 1.5);
  const Eigen::Vector3d obstacle(2.5, 0.5, 1.5);
  observeHitTwice(grid, sensor, obstacle);

  const Eigen::Vector3d range(10.0, 10.0, 10.0);
  const Eigen::Vector3d beyond(4.5, 0.5, 1.5);
  grid.integrateCloud(sensor, {beyond}, range, 3 * kSecond);
  grid.integrateCloud(sensor, {beyond}, range, 4 * kSecond);

  EXPECT_EQ(grid.getOccupancy(obstacle), 0);
  EXPECT_EQ(grid.getInflatedOccupancy(obstacle), 0);
}

TEST(RollingLogOddsGrid, ExpiryRemovesUnobservedObstacle)
{
  RollingLogOddsGrid grid(testConfig());
  const Eigen::Vector3d obstacle(2.5, 0.5, 1.5);
  observeHitTwice(grid, Eigen::Vector3d(0.5, 0.5, 1.5), obstacle);

  grid.expire(33 * kSecond);

  EXPECT_EQ(grid.getOccupancy(obstacle), 0);
  EXPECT_EQ(grid.getInflatedOccupancy(obstacle), 0);
}

TEST(RollingLogOddsGrid, RecenterPreservesOverlapAndDropsDepartedCells)
{
  RollingLogOddsGrid grid(testConfig());
  const Eigen::Vector3d sensor(0.5, 0.5, 1.5);
  const Eigen::Vector3d retained(2.5, 0.5, 1.5);
  // Keep the second ray separate from retained so that it does not correctly
  // free-space-clear the retained obstacle before the recenter assertion.
  const Eigen::Vector3d departed(-4.5, 4.5, 1.5);
  observeHitTwice(grid, sensor, retained);
  observeHitTwice(grid, sensor, departed);

  EXPECT_TRUE(grid.recenterIfNeeded(Eigen::Vector3d(4.5, 0.5, 1.5)));
  EXPECT_EQ(grid.getOccupancy(retained), 1);
  EXPECT_EQ(grid.getOccupancy(departed), -1);
}

TEST(RollingLogOddsGrid, VerticalRecenterTracksTheSensorAltitude)
{
  RollingLogOddsGrid grid(testConfig());
  EXPECT_FALSE(grid.recenterIfNeeded(Eigen::Vector3d(0.5, 0.5, 2.5)));
  EXPECT_TRUE(grid.recenterIfNeeded(Eigen::Vector3d(0.5, 0.5, 3.5)));
  EXPECT_NEAR(grid.origin().z(), 2.0, 1e-9);
}

TEST(RollingLogOddsGrid, VirtualCeilingAndOutOfBoundsAreBlocked)
{
  auto config = testConfig();
  config.virtual_ceiling_height = 3.0;
  RollingLogOddsGrid grid(config);

  EXPECT_EQ(grid.getInflatedOccupancy(Eigen::Vector3d(0.5, 0.5, 2.5)), 1);
  EXPECT_EQ(grid.getInflatedOccupancy(Eigen::Vector3d(8.0, 0.0, 1.0)), -1);
}
}  // namespace
