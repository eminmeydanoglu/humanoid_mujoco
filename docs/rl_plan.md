# G1 Locomotion RL — Uygulama Planı

Unitree G1 robotuna MuJoCo simülasyonunda velocity-tracked yürüyüş öğretmek.
Yöntem: PPO ile Reinforcement Learning.
Hedef: `(vx, vy, yaw_rate)` komutlarını takip eden stabil bir locomotion policy.

---

## Mimari Kararlar

| Konu | Karar | Gerekçe |
|---|---|---|
| Backend | Standart MuJoCo + gymnasium | GPU gerekmez, kurulumu basit |
| Kontrol | Pozisyon actuator | g1.xml değişmez, mevcut kodla tutarlı |
| PPO | stable-baselines3 | Test edilmiş, gymnasium ile doğrudan entegre |
| Paralel env | `SubprocVecEnv` | CPU multiprocessing, 16-32 env |
| Python | 3.11+ | Repo standardı |


---

## Dosya Yapısı

```
humanoid_mujoco/
├── config/
│   └── g1_config.py          # G1Config dataclass
├── envs/
│   ├── __init__.py
│   └── g1_env.py             # gymnasium.Env
├── rewards/
│   ├── __init__.py
│   └── reward_functions.py   # 13 reward terimi
├── teleop/                    # MEVCUT — dokunulmaz
└── __init__.py
train.py                       # PPO eğitim giriş noktası
play.py                        # Eğitilmiş policy görselleştirme
```

---

## Bağımlılıklar

```bash
uv add gymnasium numpy "stable-baselines3>=2.3.0" tensorboard rich tqdm
```

```toml
# pyproject.toml
dependencies = [
    "mujoco>=3.5.0",
    "gymnasium>=1.2.3",
    "numpy>=2.4.2",
    "stable-baselines3>=2.3.0",
    "tensorboard>=2.20.0",
    "rich>=15.0.0",
    "tqdm>=4.67.3",
]
```

---

## Modül 1: Konfigürasyon — `humanoid_mujoco/config/g1_config.py`

`from __future__ import annotations` + `@dataclass(slots=True)`.

```python
@dataclass(slots=True)
class G1Config:
    # Dosya yolları
    mjcf_path: Path = ...  # mujoco_menagerie/unitree_g1/scene.xml

    # Simülasyon
    dt: float = 0.02            # 50 Hz
    gait_period: float = 0.8    # saniye
    action_scale: float = 0.1   # pozisyon hedefi ölçeği (kp=500 → max ~50 N⋅m)

    # Episode
    max_episode_steps: int = 1000   # 20 saniye (50 Hz × 1000)

    # Velocity komut aralıkları (stage 0 başlangıcı; curriculum ile genişler)
    cmd_vx_range: tuple[float, float] = (0.3, 0.8)
    cmd_vy_range: tuple[float, float] = (0.0, 0.0)
    cmd_yaw_range: tuple[float, float] = (0.0, 0.0)

    # Terminate eşikleri
    min_base_height: float = 0.3
    max_roll: float = 0.785    # 45 derece (radyan)
    max_pitch: float = 0.785

    # Hedef yükseklik
    target_base_height: float = 0.78  # metre

    # Reward ağırlıkları
    w_lin_vel: float = 2.0
    w_ang_vel: float = 1.0
    w_alive: float = 0.5
    w_orientation: float = 0.2
    w_base_height: float = 0.1
    w_lin_vel_z: float = 0.5
    w_ang_vel_xy: float = 0.05
    w_torques: float = 0.0002
    w_joint_vel: float = 0.0001
    w_action_rate: float = 0.005
    w_feet_contact: float = 1.0
    w_feet_clearance: float = 0.5
    w_soft_dof_limit: float = 1.0
    soft_dof_pos_limit_factor: float = 0.9
    vel_tracking_sigma: float = 0.25

    # Domain randomization
    friction_range: tuple[float, float] = (0.3, 1.5)
    mass_offset_range: tuple[float, float] = (-2.0, 2.0)
    push_enabled: bool = False
    push_interval_steps: int = 300
    push_force_range: float = 50.0    # Newton
```

---

## Modül 2: Reward Fonksiyonları — `humanoid_mujoco/rewards/reward_functions.py`

Her fonksiyon `(model, data, config, cmd, action, last_action, phase)` alır, `float` döndürür.

| # | Fonksiyon | Ağırlık | Formül |
|---|---|---|---|
| 1 | `reward_lin_vel_tracking` | +2.0 | exp(-‖cmd_xy − vel_xy‖² / σ²) |
| 2 | `reward_ang_vel_tracking` | +1.0 | exp(-(cmd_yaw − ang_vel_z)² / σ²) |
| 3 | alive bonus | +0.5 | sabit |
| 4 | `reward_feet_contact_timing` | +1.0 | gait clock ile ayak kontağı senkronizasyonu |
| 5 | `reward_feet_clearance` | +0.5 | swing fazında yerden yükseklik |
| 6 | `penalty_orientation` | −0.2 | gravity_x² + gravity_y² |
| 7 | `penalty_base_height` | −0.1 | (base_z − 0.78)² |
| 8 | `penalty_lin_vel_z` | −0.5 | base_vel_z² |
| 9 | `penalty_ang_vel_xy` | −0.05 | ang_vel_x² + ang_vel_y² |
| 10 | `penalty_torques` | −2e-4 | Σ(data.actuator_force²) |
| 11 | `penalty_joint_vel` | −1e-4 | Σ(qvel[6:]²) |
| 12 | `penalty_action_rate` | −0.005 | Σ((action − last_action)²) |
| 13 | `penalty_soft_dof_pos_limit` | −1.0 | Σ max(0, \|normalized\| − 0.9)² |

`compute_reward_terms()` tüm terimleri dict olarak döndürür (TensorBoard'a kaydedilir).
`compute_reward()` ağırlıklı toplamı döndürür.

---

## Modül 3: Gymnasium Ortamı — `humanoid_mujoco/envs/g1_env.py`

### Observation Vektörü

| Alan | Boyut | Kaynak |
|---|---|---|
| `base_lin_vel` | 3 | `data.qvel[0:3]` → body frame |
| `base_ang_vel` | 3 | `data.qvel[3:6]` → body frame |
| `projected_gravity` | 3 | gravity rotated to body frame |
| `velocity_command` | 3 | rastgele örneklenen hedef |
| `joint_positions` | `model.nv-6` | `qpos[7:] - default_qpos[7:]` |
| `joint_velocities` | `model.nv-6` | `qvel[6:]` |
| `last_action` | `model.nu` | önceki adım ctrl |
| `clock_signal` | 2 | `[sin(phase), cos(phase)]` |

**Toplam:** `14 + 2*(model.nv-6) + model.nu` boyut (G1 için ~102)

### `step(action)`

```python
ctrl = base_ctrl + config.action_scale * action
ctrl = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
data.ctrl[:] = ctrl
mujoco.mj_step(model, data)
phase = (phase + 2 * pi * dt / gait_period) % (2 * pi)
```

### Terminate Koşulları

```python
terminated = (
    base_height < config.min_base_height
    or abs(roll)  > config.max_roll
    or abs(pitch) > config.max_pitch
)
truncated = (step_count >= config.max_episode_steps)
```

### Domain Randomization (her `reset()`'te)

```python
model.geom_friction[:, 0] *= rng.uniform(*config.friction_range)
model.body_mass[base_id]  += rng.uniform(*config.mass_offset_range)
data.qpos += rng.normal(0, 0.02, data.qpos.shape)
data.qvel += rng.normal(0, 0.1,  data.qvel.shape)
```

Push disturbance her `push_interval_steps` adımda `step()` içinde uygulanır (`push_enabled=True` ise).

---

## Modül 4: Eğitim — `train.py`

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

n_steps = 512
base_env = VecMonitor(SubprocVecEnv([make_env(config)] * n_envs))
env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[512, 256, 128], activation_fn=nn.ELU, log_std_init=-1.0),
    learning_rate=lambda f: 3e-4 * f,   # lineer azalma
    n_steps=n_steps,                     # 512 adım (tam gait döngüsü için)
    batch_size=n_envs * n_steps // 4,    # 4 minibatch
    n_epochs=5,
    clip_range=0.2,
    clip_range_vf=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.05,
    target_kl=0.01,
    vf_coef=1.0,
    max_grad_norm=1.0,
    tensorboard_log="./logs/",
)
model.learn(total_timesteps=50_000_000, callback=CallbackList([...]))
```

### Callback'ler

| Callback | Görev |
|---|---|
| `CheckpointCallback` | Her 1M adımda `.zip` kaydeder |
| `VecNormalizeSaveCallback` | Her 1M adımda `_vecnorm.pkl` kaydeder |
| `CurriculumCallback` | 50k adımlık pencerede 3 ardışık kontrol, eşiği geçince sonraki aşamaya |
| `RewardTermLogger` | Her rollout'ta 13 reward terimini TensorBoard'a kaydeder |

### Curriculum (4 Aşama)

| Aşama | vx (m/s) | vy (m/s) | yaw (rad/s) | Push |
|---|---|---|---|---|
| 0 | 0.3 → 0.8 | 0.0 | 0.0 | — |
| 1 | −0.3 → 0.8 | ±0.2 | ±0.3 | — |
| 2 | −0.3 → 0.8 | ±0.3 | ±0.5 | — |
| 3 | −0.3 → 0.8 | ±0.3 | ±0.5 | ✓ |

Geçiş: `ep_rew_mean > 5.0` koşulu 3 ardışık 50k-adım penceresinde sağlanınca sonraki aşamaya geçilir.

---

## Modül 5: Görselleştirme — `play.py`

`g1_keyboard.py`'nin GLFW viewer loop'u ve key binding'leri yeniden kullanılır.
`G1TeleopController.compute_ctrl()` yerine PPO policy `predict()` çağrısı geçer.

```bash
uv run python play.py --policy g1_policy.zip
uv run python play.py --policy g1_policy.zip --headless-steps 500
```

---

## CLI Script'leri (`pyproject.toml`)

```toml
[project.scripts]
g1-teleop = "humanoid_mujoco.teleop.g1_keyboard:main"
g1-train  = "train:main"
g1-play   = "play:main"
```

---

## Doğrulama

```bash
# Bağımlılıklar
uv run python -c "import gymnasium; import stable_baselines3"

# Ortam testi
uv run pytest tests/test_g1_env.py

# Kısa eğitim (~saniyeler)
uv run python train.py --timesteps 10000 --n-envs 4

# Policy smoke testi
uv run python play.py --policy g1_policy.zip --headless-steps 500

# Mevcut testler kırılmadı mı?
uv run pytest
```

### `tests/test_g1_env.py` — Test Dosyası

- `test_env_reset_returns_valid_obs` — obs shape ve dtype kontrolü
- `test_observation_space_matches_obs` — gymnasium space bounds check
- `test_env_step_zero_action` — sıfır aksiyonla adım, sonuçlar finite mi
- `test_env_step_random_actions` — rastgele aksiyonlarla 50 adım
- `test_env_terminate_on_fall` — `data.qpos` manipüle edip terminate assert
- `test_action_space_bounds` — aksiyon uzayı [-1, 1] ve doğru boyut
- `test_config_update_changes_command_range` — `update_config()` ile curriculum geçişi
