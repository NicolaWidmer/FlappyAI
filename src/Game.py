import pygame
import random
import time
import subprocess
import os

pygame.init()
pygame.font.init()

WIDTH = 800
HEIGHT = 600

PIPE_VELOCITY = -10

g = 8
MAX_SPEED = 40
SPEED_AFTER_ACTION = -20

FONT = pygame.font.SysFont('Comic Sans MS', 30)



class Bird:
    
    def __init__(self):
        x = WIDTH/8
        min_y = HEIGHT/8
        y = random.randint(min_y,HEIGHT - min_y)

        self.size = 10
        self.v = 0
        self.color = (0,0,0)
        self.rect = pygame.Rect(x - self.size/2, y - self.size/2, self.size, self.size)

    def draw(self,screen:pygame.Surface):
        pygame.draw.rect(screen, self.color, self.rect)

    def move(self):
        self.rect.y += self.v
        self.v += g
        self.v = min(self.v,MAX_SPEED)


class Pipe:
    
    def __init__(self):
        gap_size = 150
        width = 10

        x = WIDTH
        min_y = 50
        y = random.randint(min_y,HEIGHT-min_y-gap_size)

        self.has_passed = False
        self.v = PIPE_VELOCITY
        self.color = (0,0,255)
        self.bottom_rect = pygame.Rect(x - width/2, y + gap_size, width, HEIGHT - y - gap_size)
        self.top_rect = pygame.Rect(x - width/2, 0, width, y)

    def draw(self,screen:pygame.Surface):

        pygame.draw.rect(screen, self.color, self.bottom_rect)
        
        pygame.draw.rect(screen, self.color, self.top_rect)

    def move(self):
        self.bottom_rect.x += self.v
        self.top_rect.x += self.v


    def collide(self,bird:Bird):
        return self.top_rect.colliderect(bird.rect) or self.bottom_rect.colliderect(bird.rect)
    
    def passed(self,bird:Bird):
        return self.top_rect.x + self.top_rect.width < bird.rect.x
    
    def is_offscreen(self):
        return self.top_rect.x + self.top_rect.width < 0



class Game:

    def __init__(self):
        self.bird:Bird = Bird()
        self.pipes:list[Pipe] = [Pipe()]

        self.rounds:int = 0
        self.score:int = 0

    def reset(self):
        self.bird:Bird = Bird()
        self.pipes:list[Pipe] = [Pipe()]

        self.rounds:int = 0
        self.score:int = 0


    def step(self,action):

        
        self.bird.v =int(SPEED_AFTER_ACTION * action + self.bird.v * (1 - action))

        self.bird.move()

        for pipe in self.pipes:

            pipe.move()

            if pipe.passed(self.bird) and not pipe.has_passed:
                self.score += 1
                pipe.has_passed = True

        for pipe in self.pipes:
                if pipe.is_offscreen():
                    self.pipes.remove(pipe)

        self.rounds += 1


        if self.rounds%(WIDTH//abs(PIPE_VELOCITY)//3) == 0:
            self.pipes.append(Pipe())

        crashed = any([pipe.collide(self.bird) for pipe in self.pipes]) or self.bird.rect.y + self.bird.size > HEIGHT or self.bird.rect.y < 0

        reward = 0
        if crashed:
            reward = -1000
        else:
            reward = 100

        return self.get_state(),reward,crashed


    def get_state(self):
        return (self.bird.rect.y, self.bird.v, self.pipes[0].bottom_rect.x, self.pipes[0].bottom_rect.y)
        

    def draw(self,screen:pygame.Surface):
        
        self.bird.draw(screen)
        for pipe in self.pipes:
            pipe.draw(screen)

        score_text = FONT.render(f'Score: {self.score}', False, (0, 0, 0))
        screen.blit(score_text, (WIDTH - score_text.get_width() - 15,10))

    def run(self,actor,framerate:int = 30):

        self.reset()

        screen = pygame.display.set_mode([WIDTH, HEIGHT])

        clock = pygame.time.Clock()

        running = True
        time.sleep(2)
        while running:

            clock.tick(framerate)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((255, 255, 255))

            _, _, crashed = self.step(actor(self.get_state()))

            self.draw(screen)
        
            pygame.display.flip()

            if crashed:
                time.sleep(2)
                self.reset()

        pygame.display.quit()

    def make_video(self,actor,video_filename,max_numbers_frames:int = 1000):

        self.reset()

        frame_count = 0

        screen = pygame.display.set_mode([WIDTH, HEIGHT])

        crashed = False

        while not crashed and frame_count < max_numbers_frames:
    
            frame_count += 1
            filename = f"mp4s/screen_{frame_count:04d}.png"

            screen.fill((255, 255, 255))

            _, _, crashed = self.step(actor(self.get_state()))

            self.draw(screen)

            pygame.image.save( screen, filename )

        pygame.display.quit()

        subprocess.run(f"ffmpeg -r 30 -f image2 -s {WIDTH}x{HEIGHT} -i mp4s/screen_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {video_filename}.mp4",
                       shell=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)

        for i in range(frame_count):
            os.remove(f"mp4s/screen_{i+1:04d}.png")




if __name__ == '__main__':
    game = Game()
    actor = lambda x: 1 if pygame.key.get_pressed()[pygame.K_SPACE] else 0
    game.run(actor)