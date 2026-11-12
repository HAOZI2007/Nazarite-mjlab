#pragma once

#include <array>

namespace nazarite::xz1::him::config {

// This parameter set matches the XZ1-HIM export for the recorded training run.
// Edit only after confirming a matching training/export configuration.
inline constexpr float kControlDt = 0.02F;
inline constexpr float kPhasePeriod = 0.6F;

inline constexpr std::array<float, 12> kDefaultJointPosition{
    0, .83F, -1.7404F, 0, .83F, -1.7404F,
    0, .83F, -1.7404F, 0, .83F, -1.7404F};
inline constexpr std::array<float, 12> kActionOffset{
    0, .8483F, -1.5640F, 0, .8481F, -1.5624F,
    0, .8478F, -1.5427F, 0, .8480F, -1.5450F};
inline constexpr std::array<float, 12> kActionScale{
    .25F, .25F, .25F, .25F, .25F, .25F,
    .25F, .25F, .25F, .25F, .25F, .25F};
inline constexpr std::array<float, 12> kKp{
    32, 32, 35, 32, 32, 35, 32, 32, 35, 32, 32, 35};
inline constexpr std::array<float, 12> kKd{
    1.9F, 1.9F, 1.8F, 1.9F, 1.9F, 1.8F,
    1.9F, 1.9F, 1.8F, 1.9F, 1.9F, 1.8F};
inline constexpr std::array<float, 12> kJointLower{
    -.523599F, -1.570796F, -2.443461F, -1.047198F, -1.570796F, -2.443461F,
    -1.047198F, -1.570796F, -2.443461F, -.523599F, -1.570796F, -2.443461F};
inline constexpr std::array<float, 12> kJointUpper{
    1.047198F, 2.268928F, -.698132F, .523599F, 2.268928F, -.698132F,
    .523599F, 2.617994F, -.698132F, 1.047198F, 2.617994F, -.698132F};

}  // namespace nazarite::xz1::him::config
