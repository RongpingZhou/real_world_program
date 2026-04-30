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

# python train.py --algo dqn --env AsterixNoFrameskip-v4 --seed 37 --eval-freq -1 --n-timesteps 10000000 --tensorboard-log ./runs/

# python code_065_dqn_agent_for_all_games_with_or_without_sensor_clean_version.py \
#     --gym-id "AsterixNoFrameskip-v4" \
#     --training 1 \
#     --checkpoint-dir "./checkpoints" \
#     --cuda True \
#     --seed 37

# cd checkpoints/

# rm -rf *.pkl

# cd ..

# python train.py --algo dqn --env AsteroidsNoFrameskip-v4 --seed 37 --eval-freq -1 --n-timesteps 10000000 --tensorboard-log ./runs/

# python code_065_dqn_agent_for_all_games_with_or_without_sensor_clean_version.py \
#     --gym-id "AsteroidsNoFrameskip-v4" \
#     --training 1 \
#     --checkpoint-dir "./checkpoints" \
#     --cuda True \
#     --seed 37

# cd checkpoints/

# rm -rf *.pkl

# cd ..

python train.py --algo dqn --env GravitarNoFrameskip-v4 --seed 37 --eval-freq -1 --n-timesteps 10000000 --tensorboard-log ./runs/

python code_067_dqn_agent_for_all_games_with_or_without_sensor_clean_version.py \
    --gym-id "GravitarNoFrameskip-v4" \
    --training 1 \
    --checkpoint-dir "./checkpoints" \
    --cuda True \
    --seed 37

cd checkpoints/

rm -rf *.pkl

cd ..