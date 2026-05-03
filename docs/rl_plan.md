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
│   └── reward_functions.py   # 11 reward terimi
├── teleop/                    # MEVCUT — dokunulmaz
└── __init__.py
train.py                       # PPO eğitim giriş noktası
play.py                        # Eğitilmiş policy görselleştirme
```

---

## Bağımlılıklar

```bash
uv add gymnasium numpy "stable-baselines3>=2.3.0"
```

```toml
# pyproject.toml
dependencies = [
    "mujoco>=3.5.0",
    "gymnasium>=1.0.0",
    "numpy>=1.26",
    "stable-baselines3>=2.3.0",
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
    action_scale: float = 0.25  # pozisyon hedefi ölçeği

    # Episode
    max_episode_steps: int = 1000   # 20 saniye (50 Hz × 1000)

    # Velocity komut aralıkları
    cmd_vx_range: tuple[float, float] = (-0.3, 0.8)
    cmd_vy_range: tuple[float, float] = (-0.3, 0.3)
    cmd_yaw_range: tuple[float, float] = (-0.5, 0.5)

    # Terminate eşikleri
    min_base_height: float = 0.3
    max_roll: float = 0.785    # 45 derece (radyan)
    max_pitch: float = 0.785

    # Reward ağırlıkları
    w_lin_vel: float = 2.0
    w_ang_vel: float = 1.0
    w_alive: float = 0.5
    w_orientation: float = 0.2
    w_base_height: float = 0.1
    w_torques: float = 0.0002
    w_action_rate: float = 0.05
    w_feet_contact: float = 0.3
    w_feet_clearance: float = 0.2
    vel_tracking_sigma: float = 0.25

    # Domain randomization
    friction_range: tuple[float, float] = (0.3, 1.5)
    mass_offset_range: tuple[float, float] = (-2.0, 2.0)
    push_interval_steps: int = 300
    push_force_range: float = 50.0    # Newton
```

---

## Modül 2: Reward Fonksiyonları — `humanoid_mujoco/rewards/reward_functions.py`

Her fonksiyon `(data, model, config, cmd, last_action, phase)` alır, `float` döndürür.

| # | Fonksiyon | Formül |
|---|---|---|
| 1 | `reward_lin_vel_tracking` | exp(-‖cmd_xy − vel_xy‖² / σ²) |
| 2 | `reward_ang_vel_tracking` | exp(-(cmd_yaw − ang_vel_z)² / σ²) |
| 3 | `penalty_lin_vel_z` | base_vel_z² |
| 4 | `penalty_ang_vel_xy` | ang_vel_x² + ang_vel_y² |
| 5 | `penalty_orientation` | gravity_x² + gravity_y² |
| 6 | `penalty_base_height` | (base_z − 0.78)² |
| 7 | `penalty_torques` | Σ(data.actuator_force²) |
| 8 | `penalty_joint_vel` | Σ(qvel[6:]²) |
| 9 | `penalty_action_rate` | Σ((action − last_action)²) |
| 10 | `reward_feet_contact_timing` | gait clock ile ayak kontağı senkronizasyonu |
| 11 | `reward_feet_clearance` | swing fazında min foot clearance (0.05 m) |

`compute_reward()` tüm terimleri config ağırlıklarıyla toplar.

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

Push disturbance her `push_interval_steps` adımda `step()` içinde uygulanır.

---

## Modül 4: Eğitim — `train.py`

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

envs = SubprocVecEnv([make_env(config)] * n_envs)
model = PPO(
    "MlpPolicy",
    envs,
    policy_kwargs=dict(net_arch=[512, 256, 128], activation_fn=nn.ELU),
    learning_rate=3e-4,
    n_steps=24,
    n_epochs=5,
    clip_range=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    max_grad_norm=1.0,
    tensorboard_log="./logs/",
)
model.learn(total_timesteps=50_000_000)
model.save("g1_policy")
```

### Curriculum (4 Aşama)

| Aşama | vx_max | vy_max | yaw_max | Push |
|---|---|---|---|---|
| 0 | 0.3 | 0.0 | 0.0 | — |
| 1 | 0.8 | 0.2 | 0.3 | — |
| 2 | 0.8 | 0.3 | 0.5 | — |
| 3 | 0.8 | 0.3 | 0.5 | ✓ |

Geçiş: son 100 episode ortalama reward > 5.0 ise sonraki aşamaya.

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

### `tests/test_g1_env.py` — Yeni Test Dosyası

- `test_env_reset_returns_valid_obs` — obs shape ve dtype kontrolü
- `test_env_step_zero_action` — sıfır aksiyonla adım, sonuçlar finite mi
- `test_env_terminate_on_fall` — `data.qpos` manipüle edip terminate assert
- `test_observation_space_matches_obs` — gymnasium space bounds check
