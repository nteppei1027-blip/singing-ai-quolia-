import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Simple text-to-vector conversion
# -------------------------------
def text_to_vector(text: str) -> np.ndarray:
    return np.array([ord(c) % 32 for c in text], dtype=float)

# -------------------------------
# PyramidTensor for state propagation
# -------------------------------
class PyramidTensor:
    def __init__(self, cap=256):
        self.vertex = None
        self.history = []
        self._cap = cap

    def update_vertex(self, vector):
        self.vertex = np.asarray(vector, dtype=float)

    def propagate(self, factor=0.01):
        if self.vertex is None: return None
        v = self.vertex.copy()
        if self.history:
            v += factor * np.mean(np.vstack(self.history), axis=0)
        self.vertex = v
        return v

    def store_vector(self, vector):
        self.history.append(np.asarray(vector, dtype=float))
        if len(self.history) > self._cap:
            self.history.pop(0)

# -------------------------------
# InternalReward simplified
# -------------------------------
class InternalReward:
    def __init__(self, target_entropy=3.5, target_complexity=0.35, ema=0.2, auto_w=0.95):
        self.targets = {"entropy": target_entropy, "complexity": target_complexity}
        self.ema = ema
        self.auto_w = auto_w
        self.metrics_history = []

    def evaluate(self, tensor):
        # Entropy H
        bins = max(10, min(25, int(len(tensor)/5)))
        p, _ = np.histogram(tensor, bins=bins, density=True)
        p /= (p.sum() + 1e-12)
        H = -np.sum(p * np.log(p + 1e-12))
        
        # Complexity C
        if len(tensor) < 3:
            C = 0.0
        else:
            d2 = np.diff(np.diff(tensor))
            abs_mean = np.mean(np.abs(tensor)) + 1e-12
            C = float(np.mean(np.abs(d2)) / abs_mean)
        
        harmonic = 2.0 / (1e-12 + (1.0/(np.exp(-0.5*((H-self.targets["entropy"])/0.6)**2)) +
                                   1.0/(np.exp(-0.5*((C-self.targets["complexity"])/0.12)**2))))
        final_reward = harmonic

        if self.metrics_history:
            past_H = self.metrics_history[-1]['H']
            past_C = self.metrics_history[-1]['C']
            delta_H = H - past_H
            delta_C = C - past_C
            change_bonus = 0.5 * (abs(delta_H) + abs(delta_C))
            final_reward *= (1 + change_bonus)

        self.metrics_history.append({"H":H, "C":C})
        return float(final_reward), {"H":H, "C":C}

    def adapt_targets(self, metrics):
        for k, v in [("entropy","H"), ("complexity","C")]:
            mu_old = self.targets[k]
            mu_new = (1-self.ema)*mu_old + self.ema*metrics[v]
            self.targets[k] = self.auto_w*mu_new + (1-self.auto_w)*mu_old

# -------------------------------
# Simulation
# -------------------------------
poem = "Words are light touching all in silent glow"
poem_tensor = text_to_vector(poem)
words = poem.split()
pyramid = PyramidTensor()
ir = InternalReward()
length = len(poem_tensor)
history = []

iterations = 10
for i in range(iterations):
    # propagate tensor
    pyramid.update_vertex(poem_tensor)
    updated_vertex = pyramid.propagate()
    pyramid.store_vector(updated_vertex)

    # calculate reward metrics
    normalized = (updated_vertex - np.mean(updated_vertex)) / (np.std(updated_vertex) + 1e-12)
    reward, metrics = ir.evaluate(normalized)
    ir.adapt_targets(metrics)

    # feedback poem (top words)
    top_words = words[:min(5,len(words))]
    feedback_poem = " ".join(top_words)

    history.append({"iteration": i+1, "R":reward, "H":metrics["H"], "C":metrics["C"], "feedback": feedback_poem})
    print(f"[Iter {i+1}] R={reward:.4f} | H={metrics['H']:.4f} | C={metrics['C']:.4f} | Feedback: {feedback_poem}")

# -------------------------------
# Plot results
# -------------------------------
iters = [h['iteration'] for h in history]
R_vals = [h['R'] for h in history]
H_vals = [h['H'] for h in history]
C_vals = [h['C'] for h in history]

plt.plot(iters, R_vals, marker='o', color='red', label='Internal Reward R')
plt.plot(iters, H_vals, marker='s', color='blue', label='Entropy H')
plt.plot(iters, C_vals, marker='^', color='green', label='Complexity C')
plt.xlabel('Iteration')
plt.ylabel('Metrics')
plt.title('Singing AI-Quolia v4.1 (Simulation)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# License & Usage

This project is released for **non-commercial, educational, and research purposes only**.  
No copyright claims are asserted. You are free to study, use, and share the code and ideas.  

**Commercial use is strictly prohibited.**  

Attribution is **appreciated but not required**. If you reference or share this project, mentioning the source helps others discover it, but it is not mandatory.  

**Derivatives and modifications are allowed** under the same non-commercial terms.
