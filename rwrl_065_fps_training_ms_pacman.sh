# ALGOS: dict[str, type[BaseAlgorithm]] = {
#     "a2c": A2C,
#     "ddpg": DDPG,
#     "dqn": DQN,
#     "ppo": PPO,
#     "sac": SAC,
#     "td3": TD3,
#     # SB3 Contrib,
#     "ars": ARS,
#     "crossq": CrossQ,
#     "qrdqn": QRDQN,
#     "tqc": TQC,
#     "trpo": TRPO,
#     "ppo_lstm": RecurrentPPO,
# }

# AlienNoFrameskip-v4
# AmidarNoFrameskip-v4
# AssaultNoFrameskip-v4
# AsterixNoFrameskip-v4
# AsteroidsNoFrameskip-v4
# AtlantisNoFrameskip-v4
# BankHeistNoFrameskip-v4
# BattleZoneNoFrameskip-v4
# BeamRiderNoFrameskip-v4
# BerzerkNoFrameskip-v4
# BowlingNoFrameskip-v4
# BoxingNoFrameskip-v4
# BoxingNoFrameskip-v4
# BreakoutNoFrameskip-v4
# CentipedeNoFrameskip-v4
# ChopperCommandNoFrameskip-v4
# CrazyClimberNoFrameskip-v4
# DefenderNoFrameskip-v4
# DemonAttackNoFrameskip-v4
# DoubleDunkNoFrameskip-v4
# EnduroNoFrameskip-v4
# FishingDerbyNoFrameskip-v4
# FreewayNoFrameskip-v4
# FrostbiteNoFrameskip-v4
# GopherNoFrameskip-v4
# GravitarNoFrameskip-v4
# HeroNoFrameskip-v4
# IceHockeyNoFrameskip-v4
# JamesbondNoFrameskip-v4
# KangarooNoFrameskip-v4
# KrullNoFrameskip-v4
# KungFuMasterNoFrameskip-v4
# MontezumaRevengeNoFrameskip-v4
# MsPacmanNoFrameskip-v4
# NameThisGameNoFrameskip-v4
# PhoenixNoFrameskip-v4
# PitfallNoFrameskip-v4
# PongNoFrameskip-v4
# PrivateEyeNoFrameskip-v4
# QbertNoFrameskip-v4
# RiverraidNoFrameskip-v4
# RoadRunnerNoFrameskip-v4
# RobotankNoFrameskip-v4
# SeaquestNoFrameskip-v4
# SkiingNoFrameskip-v4
# SolarisNoFrameskip-v4
# SpaceInvadersNoFrameskip-v4
# StarGunnerNoFrameskip-v4
# TennisNoFrameskip-v4
# TimePilotNoFrameskip-v4
# TutankhamNoFrameskip-v4
# UpNDownNoFrameskip-v4
# VentureNoFrameskip-v4
# VideoPinballNoFrameskip-v4
# WizardOfWorNoFrameskip-v4
# YarsRevengeNoFrameskip-v4
# ZaxxonNoFrameskip-v4

python code_065_dqn_agent_for_real_world_system_ms_pacman_at_60fps.py \
    --gym-id "MsPacmanNoFrameskip-v4" \
    --max-episode-steps 60000 \
    --checkpoint-dir "./checkpoints" \
    --training 1 \
    --test 0 \
    --cuda False \
    --fps 60 \
    --seed 37 \
    --crop 1