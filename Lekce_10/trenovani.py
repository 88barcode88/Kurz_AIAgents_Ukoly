
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 1. Definice hyperparametrů
NUM_EPISODES = 20000  # Počet epizod pro trénink
LEARNING_RATE = 0.1  # Learning rate (alpha)
DISCOUNT_FACTOR = 0.99  # Discount factor (gamma)
EPSILON_START = 1.0  # Počáteční hodnota epsilon pro exploraci
EPSILON_MIN = 0.01  # Minimální hodnota epsilon
EPSILON_DECAY = 0.0001  # Míra snižování epsilon

# 2. Inicializace prostředí
env = gym.make('FrozenLake-v1', is_slippery=False)
n_states = env.observation_space.n  # Počet stavů (16 pro FrozenLake 4x4)
n_actions = env.action_space.n  # Počet akcí (4: nahoru, dolů, doleva, doprava)

print(f"Počet stavů: {n_states}")
print(f"Počet akcí: {n_actions}")

# 3. Inicializace Q-tabulky
# Q-tabulka má rozměry [počet_stavů x počet_akcí]
q_table = np.zeros((n_states, n_actions))

# 4. Funkce pro epsilon-greedy výběr akce
def epsilon_greedy_action(state, epsilon):
    """
    Vybere akci pomocí epsilon-greedy strategie.
    
    Args:
        state: Aktuální stav
        epsilon: Pravděpodobnost náhodné akce (explorace)
    
    Returns:
        Vybraná akce
    """
    if np.random.random() < epsilon:
        # Explorace: náhodná akce
        return env.action_space.sample()
    else:
        # Exploitace: nejlepší akce podle Q-tabulky
        return np.argmax(q_table[state])

# 5. Hlavní trénovací smyčka
episode_rewards = []  # Seznam pro ukládání celkových odměn
epsilon = EPSILON_START

print("\nZačíná trénink Q-learning agenta...")

for episode in range(NUM_EPISODES):
    # Reset prostředí na začátku každé epizody
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    # Smyčka pro jednu epizodu
    while not done:
        # Výběr akce pomocí epsilon-greedy strategie
        action = epsilon_greedy_action(state, epsilon)
        
        # Provedení akce v prostředí
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Aktualizace Q-hodnoty pomocí Bellmanovy rovnice
        # Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        old_q_value = q_table[state, action]
        next_max_q = np.max(q_table[next_state])
        new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - old_q_value)
        q_table[state, action] = new_q_value
        
        # Aktualizace stavu a celkové odměny
        state = next_state
        total_reward += reward
        
        # Kontrola, zda byla epizoda ukončena kvůli časovému limitu
        if truncated:
            done = True
    
    # Uložení celkové odměny za epizodu
    episode_rewards.append(total_reward)
    
    # Snížení epsilon (postupné snižování explorace)
    epsilon = max(EPSILON_MIN, epsilon - EPSILON_DECAY)
    
    # Výpis průběhu každých 1000 epizod
    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(episode_rewards[-100:])  # Průměr posledních 100 epizod
        print(f"Epizoda {episode + 1}/{NUM_EPISODES}, Průměrná odměna (posledních 100): {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

print("\nTrénink dokončen!")

# 6. Vyhodnocení a vizualizace

# Vytvoření složky pro výsledky, pokud neexistuje
results_dir = "vysledky"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Vykreslení grafu učení
plt.figure(figsize=(12, 6))

# Graf 1: Celkové odměny v průběhu epizod
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, alpha=0.6)
plt.title('Celkové odměny v průběhu tréninku')
plt.xlabel('Epizoda')
plt.ylabel('Celková odměna')

# Přidání klouzavého průměru pro lepší vizualizaci trendu
window_size = 100
if len(episode_rewards) >= window_size:
    moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'Klouzavý průměr ({window_size} epizod)')
    plt.legend()

# Graf 2: Histogram finálních odměn
plt.subplot(1, 2, 2)
last_1000_rewards = episode_rewards[-1000:]
plt.hist(last_1000_rewards, bins=20, edgecolor='black')
plt.title('Histogram odměn (posledních 1000 epizod)')
plt.xlabel('Odměna')
plt.ylabel('Počet epizod')

plt.tight_layout()

# Uložení grafu s časovým razítkem
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"graf_uceni_{timestamp}.png"
filepath = os.path.join(results_dir, filename)
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print(f"\nGraf uložen do: {filepath}")

# Zobrazení grafu
plt.show()

# Uložení Q-tabulky do souboru
q_table_filename = f"q_tabulka_{timestamp}.npy"
q_table_filepath = os.path.join(results_dir, q_table_filename)
np.save(q_table_filepath, q_table)
print(f"Q-tabulka uložena do: {q_table_filepath}")

# Uložení statistik do textového souboru
stats_filename = f"statistiky_{timestamp}.txt"
stats_filepath = os.path.join(results_dir, stats_filename)

with open(stats_filepath, 'w', encoding='utf-8') as f:
    f.write("Statistiky tréninku Q-learning agenta\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Počet epizod: {NUM_EPISODES}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"Discount factor: {DISCOUNT_FACTOR}\n")
    f.write(f"Epsilon start: {EPSILON_START}\n")
    f.write(f"Epsilon min: {EPSILON_MIN}\n")
    f.write(f"Epsilon decay: {EPSILON_DECAY}\n\n")
    
    f.write(f"Průměrná odměna (posledních 100 epizod): {np.mean(episode_rewards[-100:]):.3f}\n")
    f.write(f"Průměrná odměna (posledních 1000 epizod): {np.mean(episode_rewards[-1000:]):.3f}\n")
    f.write(f"Maximální dosažená odměna: {max(episode_rewards)}\n")
    f.write(f"Úspěšnost (posledních 1000 epizod): {sum(last_1000_rewards)/len(last_1000_rewards)*100:.1f}%\n\n")
    
    f.write("Finální Q-tabulka:\n")
    f.write("Stav | Nahoru | Dolů | Doleva | Doprava\n")
    f.write("-" * 45 + "\n")
    for state in range(n_states):
        f.write(f"{state:4d} | {q_table[state, 0]:6.3f} | {q_table[state, 1]:6.3f} | {q_table[state, 2]:6.3f} | {q_table[state, 3]:6.3f}\n")

print(f"Statistiky uloženy do: {stats_filepath}")

# Výpis finální Q-tabulky
print("\nFinální Q-tabulka:")
print("Stav | Nahoru | Dolů | Doleva | Doprava")
print("-" * 45)
for state in range(n_states):
    print(f"{state:4d} | {q_table[state, 0]:6.3f} | {q_table[state, 1]:6.3f} | {q_table[state, 2]:6.3f} | {q_table[state, 3]:6.3f}")

# 7. Demonstrace naučeného agenta

print("\n\nDemonstrace naučeného agenta (5 epizod):")
print("=" * 50)

# Vytvoření nového prostředí s vizualizací
demo_env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')

for demo_episode in range(5):
    state, _ = demo_env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    print(f"\nEpizoda {demo_episode + 1}:")
    
    while not done and steps < 100:  # Limit kroků pro případ nekonečné smyčky
        # Použití pouze nejlepší akce (epsilon = 0)
        action = np.argmax(q_table[state])
        
        # Provedení akce
        next_state, reward, done, truncated, _ = demo_env.step(action)
        
        total_reward += reward
        state = next_state
        steps += 1
        
        if truncated:
            done = True
    
    if total_reward > 0:
        print(f"✓ Úspěch! Agent dosáhl cíle v {steps} krocích.")
    else:
        print(f"✗ Neúspěch. Agent nedosáhl cíle.")

# Zavření prostředí
demo_env.close()
env.close()

print("\n\nProgram ukončen.")