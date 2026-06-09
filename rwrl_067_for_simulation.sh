# 1     AlienNoFrameskip-v4
# 2     AmidarNoFrameskip-v4
# 3     AssaultNoFrameskip-v4
# 4     AsterixNoFrameskip-v4
# 5     AsteroidsNoFrameskip-v4
# 6     AtlantisNoFrameskip-v4
# 7     BankHeistNoFrameskip-v4
# 8     BattleZoneNoFrameskip-v4
# 9     BeamRiderNoFrameskip-v4
# 10    BerzerkNoFrameskip-v4
# 11    BowlingNoFrameskip-v4
# 12    BoxingNoFrameskip-v4
# 13    BreakoutNoFrameskip-v4
# 14    CentipedeNoFrameskip-v4
# 15    ChopperCommandNoFrameskip-v4
# 16    CrazyClimberNoFrameskip-v4
# 17    DefenderNoFrameskip-v4
# 18    DemonAttackNoFrameskip-v4
# 19    DoubleDunkNoFrameskip-v4
# 20    EnduroNoFrameskip-v4
# 21    FishingDerbyNoFrameskip-v4
# 22    FreewayNoFrameskip-v4
# 23    FrostbiteNoFrameskip-v4
# 24    GopherNoFrameskip-v4
# 25    GravitarNoFrameskip-v4
# 26    HeroNoFrameskip-v4
# 27    IceHockeyNoFrameskip-v4
# 28    JamesbondNoFrameskip-v4
# 29    KangarooNoFrameskip-v4
# 30    KrullNoFrameskip-v4
# 31    KungFuMasterNoFrameskip-v4
# 32    MontezumaRevengeNoFrameskip-v4
# 33    MsPacmanNoFrameskip-v4
# 34    NameThisGameNoFrameskip-v4
# 35    PhoenixNoFrameskip-v4
# 36    PitfallNoFrameskip-v4
# 37    PongNoFrameskip-v4
# 38    PrivateEyeNoFrameskip-v4
# 39    QbertNoFrameskip-v4
# 40    RiverraidNoFrameskip-v4
# 41    RoadRunnerNoFrameskip-v4
# 42    RobotankNoFrameskip-v4
# 43    SeaquestNoFrameskip-v4
# 44    SkiingNoFrameskip-v4
# 45    SolarisNoFrameskip-v4
# 46    SpaceInvadersNoFrameskip-v4
# 47    StarGunnerNoFrameskip-v4
# 48    ALE/Surround-v5
# 49    TennisNoFrameskip-v4
# 50    TimePilotNoFrameskip-v4
# 51    TutankhamNoFrameskip-v4
# 52    UpNDownNoFrameskip-v4
# 53    VentureNoFrameskip-v4
# 54    VideoPinballNoFrameskip-v4
# 55    WizardOfWorNoFrameskip-v4
# 56    YarsRevengeNoFrameskip-v4
# 57    ZaxxonNoFrameskip-v4

python train.py --algo dqn --env ALE/Surround-v5 --seed 37 --eval-freq -1 --n-timesteps 10000000 --tensorboard-log ./runs/

python code_067_dqn_agent_training_for_real_world_input_system.py \
    --gym-id "ALE/Surround-v5" \
    --training 1 \
    --checkpoint-dir "./checkpoints" \
    --cuda True \
    --seed 37

cd checkpoints/

rm -rf *.pkl

cd ..

# python train.py --algo dqn --env ZaxxonNoFrameskip-v4 --seed 37 --eval-freq -1 --n-timesteps 10000000 --tensorboard-log ./runs/

# python code_067_dqn_agent_training_for_real_world_input_system.py \
#     --gym-id "ZaxxonNoFrameskip-v4" \
#     --training 1 \
#     --checkpoint-dir "./checkpoints" \
#     --cuda True \
#     --seed 37

# cd checkpoints/

# rm -rf *.pkl

# cd ..

# python train.py --algo dqn --env VentureNoFrameskip-v4 --seed 37 --eval-freq -1 --n-timesteps 10000000 --tensorboard-log ./runs/

# python code_067_dqn_agent_training_for_real_world_input_system.py \
#     --gym-id "VentureNoFrameskip-v4" \
#     --training 1 \
#     --checkpoint-dir "./checkpoints" \
#     --cuda True \
#     --seed 37

# cd checkpoints/

# rm -rf *.pkl

# cd ..