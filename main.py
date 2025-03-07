from numpy import ndarray
import numpy as np
import pygame
from pygame import Vector2
from random import randint
from agent import Agent
from helper import SliceableDeque, CELL_NUMBER, CELL_SIZE, SCREEN, CLOCK, head_images, tail_images, align_images, turn_images

pygame.init()
FONT = pygame.font.Font("./Font/PoetsenOne-Regular.ttf", 25)


def within_bounds(pos):
    return (0 <= pos.y < CELL_NUMBER and  0 <= pos.x < CELL_NUMBER)

class Game:
    def __init__(self):
        self.CLOCK = pygame.time.Clock()
        self.HIGH_SCORE = 0
        self.agent = Agent()
        
        
        self.SCREEN_UPDATE = pygame.USEREVENT
        pygame.time.set_timer(self.SCREEN_UPDATE, 1)
    
    def run(self):
                
        self.main = Main()

        while not self.main.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == self.SCREEN_UPDATE:
                    state = self.main.get_state()
                    final_move = self.agent.get_action(state)
                    reward, done, score = self.main.update(np.argmax(final_move))
                    new_state = self.main.get_state()
                    
                    self.agent.train_short_memory(state, final_move, reward, new_state, done )
                    self.agent.remember(state, final_move, reward, new_state, done)
                                            
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()


            SCREEN.fill((175, 215, 70))
            self.draw_grass()
            self.main.draw()
            self.display_score()
            CLOCK.tick(60)
            pygame.display.update()
        self.agent.train_long_memory()
        self.agent.n_games += 1
        if self.agent.n_games == 50:
            pygame.time.set_timer(self.SCREEN_UPDATE, 100)
            
        self.HIGH_SCORE = max(self.HIGH_SCORE, self.main.score)
        print(f"Game: {self.agent.n_games}, Score: {self.main.score}, High Score: {self.HIGH_SCORE}")

    def draw_grass(self):
        grass_color = (167, 209, 61)
        for row in range(CELL_NUMBER):
            for col in range(CELL_NUMBER):
                if (row + col) & 1:
                    rect = pygame.Rect(row * CELL_SIZE, col * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(SCREEN, grass_color, rect)
    
    def display_score(self):
        score_text = "Score: " + str(int(self.main.score))
        score_surf = FONT.render(score_text, True, (56, 74, 12))
        pos = CELL_NUMBER + 60
        rect = score_surf.get_rect(center = (pos,pos))
        SCREEN.blit(score_surf, rect)
class Snake:
    def __init__(self):
        self.body: SliceableDeque[Vector2] = SliceableDeque([Vector2(4,10), Vector2(3,10), Vector2(2,10)])
        self.direction = Vector2(1,0)
        self.increase_size = False
        self.head_image = head_images['right']
        self.tail_image =  tail_images['left']
    def draw_snake(self):
        for idx,cell in enumerate(self.body):
            rect = pygame.Rect(cell.x * CELL_SIZE,cell.y * CELL_SIZE,CELL_SIZE,CELL_SIZE)
            if idx == 0:
                self.update_snake_head()
                SCREEN.blit(self.head_image, rect)
            elif idx == len(self.body) - 1:
                self.update_snake_tail()
                SCREEN.blit(self.tail_image, rect)
            else:
                prev = self.body[idx - 1] - cell
                nxt = self.body[idx + 1] - cell
                if prev.x == nxt.x:
                    SCREEN.blit(align_images['vertical'], rect)
                elif prev.y == nxt.y:
                    SCREEN.blit(align_images['horizontal'], rect)
                elif nxt.y == 0:
                    if prev.y == -1:
                        if nxt.x == 1:
                            SCREEN.blit(turn_images["tr"], rect)
                        if nxt.x == -1:
                            SCREEN.blit(turn_images['tl'], rect)
                    if prev.y == 1:
                        if nxt.x == 1:
                            SCREEN.blit(turn_images["br"], rect)
                        if nxt.x == -1:
                            SCREEN.blit(turn_images['bl'], rect)
                
                elif prev.y == 0:
                    if nxt.y == -1:
                        if prev.x == 1:
                            SCREEN.blit(turn_images["tr"], rect)
                        if prev.x == -1:
                            SCREEN.blit(turn_images['tl'], rect)
                    if nxt.y == 1:
                        if prev.x == 1:
                            SCREEN.blit(turn_images["br"], rect)
                        if prev.x == -1:
                            SCREEN.blit(turn_images['bl'], rect)        
                else:
                    pygame.draw.rect(SCREEN, (12,100,166), rect)
        
    def update_snake_tail(self):
        tail_dir = self.body[-1] - self.body[-2]
        if tail_dir.x == 1:
            self.tail_image = tail_images['right']
        if tail_dir.x == -1:
            self.tail_image = tail_images['left']
        if tail_dir.y == 1:
            self.tail_image = tail_images['down']
        if tail_dir.y == -1:
            self.tail_image = tail_images['up']
            
    def update_snake_head(self):
        if self.direction.y == 1:
            self.head_image = head_images['down']
        if self.direction.y == -1:
            self.head_image = head_images['up']
        if self.direction.x == 1:
            self.head_image = head_images['right']
        if self.direction.x == -1:
            self.head_image = head_images['left']
            
    def body_around(self, pos):
        return pos in self.body[2:]
        
    def move_snake(self):
        self.body.appendleft(self.body[0] + self.direction)
        if self.increase_size:
            self.increase_size = False
            return
        self.body.pop()

    def add_block(self):
        self.increase_size = True
        
    def die(self):
        hit_body = self.body_around(self.body[0])
        if hit_body or not within_bounds(self.body[0]):
            return True
        return False
        
    

class Fruit:
    def __init__(self):
        self.x = randint(0,CELL_NUMBER-1) 
        self.y =  randint(0,CELL_NUMBER-1) 
        self.coor = Vector2(self.x, self.y)
        self.surface = pygame.image.load("./Graphics/apple.png").convert_alpha()
    
    def draw_fruit(self):
        self.rect = pygame.Rect(self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE,CELL_SIZE)        
        SCREEN.blit(self.surface, self.rect)
        
    def get_new_coor(self):
        self.x = randint(0,CELL_NUMBER-1) 
        self.y =  randint(0,CELL_NUMBER-1) 
        self.coor = Vector2(self.x, self.y)
        
class Main:
    def __init__(self):
        self.snake = Snake()
        self.fruit = Fruit()
        self.score = 0.0
        self.done = 0.0
        self.reward = -0.1
        self.frame_iter = 0
        
    
    def draw(self):
        self.snake.draw_snake()
        self.fruit.draw_fruit()
        
    def update(self, action):
        self.move(action)
        self.snake.move_snake()
        fruit_eaten = self.eat_fruit()
        self.done = float(self.snake.die())
        self.frame_iter += 1
        if self.frame_iter > 2000:
            self.done = 1
        if self.done:
            self.reward = -10.0
        elif fruit_eaten:
            self.reward = 10.0
        else:
            self.reward = -0.1
        return self.reward, self.done, self.score
    
        
        
        
    def move(self, action: ndarray):
        direction = self.snake.direction
        if action == 0:
            if direction.y != 0:
                self.snake.direction = Vector2(direction.y,0)
            elif direction.x != 0:
                self.snake.direction = Vector2(0, -direction.x)
        if action == 2:
            if direction.y != 0:
                self.snake.direction = Vector2(-direction.y,0)
            elif direction.x != 0:
                self.snake.direction = Vector2(0, direction.x)
        
            
    
    def get_state(self):
        head = self.snake.body[0]
        direction = self.snake.direction
        food_ahead = float(
            direction.y == 1 and self.fruit.y > head.y or
            direction.y == -1 and self.fruit.y < head.y or
            direction.x == 1 and self.fruit.x > head.x or
            direction.x == -1 and self.fruit.x < head.x 
        )
        food_clockwise = float(
            direction.y == 1 and self.fruit.x < head.x or
            direction.y == -1 and self.fruit.x > head.x or
            direction.x == 1 and self.fruit.y > head.y or
            direction.x == -1 and self.fruit.y < head.y 
        )
        food_anticlockwise = float(
            direction.y == 1 and self.fruit.x > head.x or
            direction.y == -1 and self.fruit.x < head.x or
            direction.x == 1 and self.fruit.y < head.y or
            direction.x == -1 and self.fruit.y > head.y 
        )
        food_behind =  float(
             direction.y == 1 and self.fruit.y < head.y or
            direction.y == -1 and self.fruit.y > head.y or
            direction.x == 1 and self.fruit.x < head.x or
            direction.x == -1 and self.fruit.x > head.x 
        )
        
        ahead = head + direction
        danger_ahead = float(not within_bounds(ahead))
        body_ahead = float(self.snake.body_around(ahead))
        
        
        
        dir_clockwise = Vector2(direction.y, -direction.x)
        dir_anticlockwise = Vector2(-direction.y, direction.x)

        clockwise = head + dir_clockwise
        anticlockwise = head + dir_anticlockwise
        danger_clockwise = float(not within_bounds(clockwise))
        danger_anticlockwise = float(not within_bounds(anticlockwise))
        body_clockwise = float(self.snake.body_around(clockwise))
        body_anticlockwise = float(self.snake.body_around(anticlockwise))
        return np.array([float(direction.y == -1) , float(direction.y == 1), float(direction.x == -1), float(direction.x == 1), food_anticlockwise, food_ahead, food_clockwise, food_behind, danger_ahead, danger_clockwise, danger_anticlockwise, body_ahead, body_clockwise, body_anticlockwise, self.frame_iter/1000], dtype=float)
        
        
    def eat_fruit(self):
        if self.snake.body[0] == self.fruit.coor:
            self.score += 1
            self.snake.add_block()
            while self.fruit.coor in self.snake.body:
                self.fruit.get_new_coor()
            return True


if __name__ == "__main__":
    game = Game()
    while True: 
        game.run()