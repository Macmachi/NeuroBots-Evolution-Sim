'''
*
* PROJET : NeuroBots-Evolution-Sim
* AUTEUR : Arnaud R.
* VERSION : 1.0.0
* Licence : MIT
*
'''

import pygame
import random
import numpy as np

# Initialisation de Pygame
pygame.init()

# Paramètres de la simulation
WIDTH, HEIGHT = 800, 600
AGENT_COUNT = 50
GENERATION_TIME = 10000  # Durée d'une génération en millisecondes

# Couleurs
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Création de la fenêtre
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulation de réseau de neurones")

# Liste pour stocker l'historique des récompenses
reward_history = []

class Agent:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.reward = 0
        self.network = NeuralNetwork()

    def move(self, center_x, center_y):
        inputs = [self.x / WIDTH, self.y / HEIGHT, center_x / WIDTH, center_y / HEIGHT]
        outputs = self.network.forward(inputs)
        dx, dy = outputs[0] * 5 - 2.5, outputs[1] * 5 - 2.5  # Déplacement entre -2.5 et 2.5
        self.x = max(0, min(WIDTH, self.x + dx))
        self.y = max(0, min(HEIGHT, self.y + dy))

    def calculate_reward(self, center_x, center_y):
        distance = ((self.x - center_x) ** 2 + (self.y - center_y) ** 2) ** 0.5
        self.reward += max(0, 1 - distance / (WIDTH / 2))

class NeuralNetwork:
    def __init__(self):
        self.w1 = np.random.randn(4, 4)
        self.w2 = np.random.randn(4, 2)

    def forward(self, x):
        x = np.array(x)
        h = np.tanh(np.dot(x, self.w1))
        return np.tanh(np.dot(h, self.w2))

def create_new_generation(agents):
    agents.sort(key=lambda x: x.reward, reverse=True)
    top_agents = agents[:AGENT_COUNT // 5]
    new_agents = []

    for _ in range(AGENT_COUNT):
        parent1, parent2 = random.sample(top_agents, 2)
        child = Agent()
        child.network.w1 = (parent1.network.w1 + parent2.network.w1) / 2
        child.network.w2 = (parent1.network.w2 + parent2.network.w2) / 2
        
        # Ajout de mutation
        child.network.w1 += np.random.randn(*child.network.w1.shape) * 0.1
        child.network.w2 += np.random.randn(*child.network.w2.shape) * 0.1
        
        new_agents.append(child)

    return new_agents

def draw_network(screen, agent, x, y, width, height):
    # Couleurs
    NODE_COLOR = (100, 100, 100)
    POSITIVE_COLOR = (0, 255, 0)
    NEGATIVE_COLOR = (255, 0, 0)
    
    # Calculer les positions des nœuds
    input_y = y + height * 0.2
    hidden_y = y + height * 0.5
    output_y = y + height * 0.8
    
    input_xs = [x + width * (i+1)/(5) for i in range(4)]
    hidden_xs = [x + width * (i+1)/(5) for i in range(4)]
    output_xs = [x + width * (i+1)/(3) for i in range(2)]
    
    # Dessiner les connexions
    for i in range(4):
        for j in range(4):
            weight = agent.network.w1[i, j]
            color = POSITIVE_COLOR if weight > 0 else NEGATIVE_COLOR
            start_pos = (int(input_xs[i]), int(input_y))
            end_pos = (int(hidden_xs[j]), int(hidden_y))
            pygame.draw.line(screen, color, start_pos, end_pos, max(1, int(abs(weight) * 3)))
    
    for i in range(4):
        for j in range(2):
            weight = agent.network.w2[i, j]
            color = POSITIVE_COLOR if weight > 0 else NEGATIVE_COLOR
            start_pos = (int(hidden_xs[i]), int(hidden_y))
            end_pos = (int(output_xs[j]), int(output_y))
            pygame.draw.line(screen, color, start_pos, end_pos, max(1, int(abs(weight) * 3)))
    
    # Dessiner les nœuds
    for x_pos in input_xs:
        pygame.draw.circle(screen, NODE_COLOR, (int(x_pos), int(input_y)), 5)
    for x_pos in hidden_xs:
        pygame.draw.circle(screen, NODE_COLOR, (int(x_pos), int(hidden_y)), 5)
    for x_pos in output_xs:
        pygame.draw.circle(screen, NODE_COLOR, (int(x_pos), int(output_y)), 5)

def draw_reward_graph(screen, x, y, width, height):
    if len(reward_history) < 2:
        return

    pygame.draw.rect(screen, WHITE, (x, y, width, height))
    pygame.draw.rect(screen, BLACK, (x, y, width, height), 1)

    max_reward = max(reward_history)
    min_reward = min(reward_history)
    range_reward = max(max_reward - min_reward, 1)

    for i in range(1, len(reward_history)):
        start_x = x + (i-1) * width / (len(reward_history) - 1)
        start_y = y + height - (reward_history[i-1] - min_reward) / range_reward * height
        end_x = x + i * width / (len(reward_history) - 1)
        end_y = y + height - (reward_history[i] - min_reward) / range_reward * height

        pygame.draw.line(screen, GREEN, (start_x, start_y), (end_x, end_y), 2)

    font = pygame.font.Font(None, 24)
    text = font.render("Reward History", True, BLACK)
    screen.blit(text, (x + width // 2 - text.get_width() // 2, y - 30))

# Création des agents initiaux
agents = [Agent() for _ in range(AGENT_COUNT)]

# Boucle principale
running = True
clock = pygame.time.Clock()
generation = 1
start_time = pygame.time.get_ticks()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Mise à jour et dessin
    screen.fill(WHITE)
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    pygame.draw.circle(screen, RED, (center_x, center_y), 10)

    for agent in agents:
        agent.move(center_x, center_y)
        agent.calculate_reward(center_x, center_y)
        pygame.draw.circle(screen, BLUE, (int(agent.x), int(agent.y)), 5)

    # Affichage du numéro de génération
    font = pygame.font.Font(None, 36)
    text = font.render(f"Generation: {generation}", True, BLACK)
    screen.blit(text, (10, 10))

    # Dessiner le réseau du premier agent
    if len(agents) > 0:
        draw_network(screen, agents[0], WIDTH - 200, HEIGHT - 300, 180, 130)

    # Dessiner le graphique des récompenses
    draw_reward_graph(screen, WIDTH - 200, HEIGHT - 150, 180, 130)

    pygame.display.flip()
    clock.tick(60)

    # Vérification du temps écoulé pour la génération
    current_time = pygame.time.get_ticks()
    if current_time - start_time > GENERATION_TIME:
        reward_history.append(agents[0].reward)  # Ajouter la récompense du premier agent à l'historique
        agents = create_new_generation(agents)
        generation += 1
        start_time = current_time

pygame.quit()