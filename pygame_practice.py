import pygame
import sys
import os
from pose_match_overlay_selfie import PoseMatchingSystem

class PoseMatchGame:
    def __init__(self):
        pygame.init()
        self.screen_width = 1280
        self.screen_height = 480
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pose Matching Game")
        
        # 색상 정의
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        
        # 폰트 초기화
        self.font = pygame.font.Font(None, 64)
        self.small_font = pygame.font.Font(None, 36)
        
        # 게임 상태
        self.running = True
        self.game_state = "menu"  # menu, playing, complete
        
    def draw_menu(self):
        self.screen.fill(self.BLACK)
        
        # 타이틀 텍스트
        title = self.font.render("Pose Matching Game", True, self.WHITE)
        title_rect = title.get_rect(center=(self.screen_width//2, self.screen_height//3))
        self.screen.blit(title, title_rect)
        
        # 시작 버튼
        start_text = self.font.render("Press SPACE to Start", True, self.GREEN)
        start_rect = start_text.get_rect(center=(self.screen_width//2, self.screen_height*2//3))
        self.screen.blit(start_text, start_rect)
        
        # 종료 안내
        quit_text = self.small_font.render("Press ESC to Quit", True, self.WHITE)
        quit_rect = quit_text.get_rect(center=(self.screen_width//2, self.screen_height - 50))
        self.screen.blit(quit_text, quit_rect)
        
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE and self.game_state == "menu":
                        self.game_state = "playing"
                        # PoseMatchingSystem 실행
                        try:
                            pose_system = PoseMatchingSystem()
                            image_path = "movies/son.jpg"  # 템플릿 이미지 경로 설정
                            pose_system.create_template(image_path)
                            pose_system.run_webcam_matching()
                        except Exception as e:
                            print(f"Error: {e}")
                        finally:
                            self.game_state = "menu"
            
            if self.game_state == "menu":
                self.draw_menu()
            
            pygame.display.flip()
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = PoseMatchGame()
    game.run()