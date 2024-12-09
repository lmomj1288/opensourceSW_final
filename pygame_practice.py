import pygame
import sys
import os
from pose_match_overlay_selfie import PoseMatchingSystem
import random

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
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        
        # 폰트 초기화
        self.font = pygame.font.Font(None, 64)
        self.small_font = pygame.font.Font(None, 36)
        
        # 게임 상태
        self.running = True
        self.game_state = "menu"  # menu, select_mode, playing_sequence, playing_random, complete
        self.selected_option = 0  # 0: Sequence Mode, 1: Random Mode
        
        # 이미지 경로 설정
        self.image_paths = [
            "C:/Users/lmomj/Desktop/opensource/final/movies/dybala.jpg",
            "C:/Users/lmomj/Desktop/opensource/final/movies/son.jpg",
            "C:/Users/lmomj/Desktop/opensource/final/movies/yan.jpg"
        ]

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

    def draw_mode_selection(self):
        self.screen.fill(self.BLACK)
        
        # 모드 선택 타이틀
        title = self.font.render("Select Game Mode", True, self.WHITE)
        title_rect = title.get_rect(center=(self.screen_width//2, self.screen_height//4))
        self.screen.blit(title, title_rect)
        
        # 시퀀스 모드 옵션
        sequence_color = self.GREEN if self.selected_option == 0 else self.WHITE
        sequence_text = self.font.render("1. Sequence Mode", True, sequence_color)
        sequence_rect = sequence_text.get_rect(center=(self.screen_width//2, self.screen_height//2))
        self.screen.blit(sequence_text, sequence_rect)
        
        # 랜덤 모드 옵션
        random_color = self.GREEN if self.selected_option == 1 else self.WHITE
        random_text = self.font.render("2. Random Mode", True, random_color)
        random_rect = random_text.get_rect(center=(self.screen_width//2, self.screen_height//2 + 80))
        self.screen.blit(random_text, random_rect)
        
        # 선택 안내
        instruction_text = self.small_font.render("Use UP/DOWN arrows and SPACE to select", True, self.BLUE)
        instruction_rect = instruction_text.get_rect(center=(self.screen_width//2, self.screen_height*3//4))
        self.screen.blit(instruction_text, instruction_rect)

    def draw_complete(self):
        self.screen.fill(self.BLACK)
        
        # 완료 메시지
        complete_text = self.font.render("All Poses Completed!", True, self.GREEN)
        complete_rect = complete_text.get_rect(center=(self.screen_width//2, self.screen_height//3))
        self.screen.blit(complete_text, complete_rect)
        
        # 메인 메뉴로 돌아가기 안내
        menu_text = self.font.render("Press SPACE for Main Menu", True, self.WHITE)
        menu_rect = menu_text.get_rect(center=(self.screen_width//2, self.screen_height*2//3))
        self.screen.blit(menu_text, menu_rect)

    def run_sequence_mode(self):
        try:
            pose_system = PoseMatchingSystem()
            print("템플릿 포즈 로딩 중...")
            pose_system.load_templates(self.image_paths)
            print("웹캠 매칭 시작...")
            pose_system.run_webcam_matching()
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def run_random_mode(self):
        try:
            random_paths = random.sample(self.image_paths, 3)  # 랜덤으로 3개 선택
            pose_system = PoseMatchingSystem()
            print("템플릿 포즈 로딩 중...")
            pose_system.load_templates(random_paths)
            print("웹캠 매칭 시작...")
            pose_system.run_webcam_matching()
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.game_state == "select_mode":
                            self.game_state = "menu"
                        else:
                            self.running = False
                    
                    elif event.key == pygame.K_SPACE:
                        if self.game_state == "menu":
                            self.game_state = "select_mode"
                        elif self.game_state == "select_mode":
                            if self.selected_option == 0:
                                self.game_state = "playing_sequence"
                                if self.run_sequence_mode():
                                    self.game_state = "complete"
                                else:
                                    self.game_state = "menu"
                            else:
                                self.game_state = "playing_random"
                                if self.run_random_mode():
                                    self.game_state = "complete"
                                else:
                                    self.game_state = "menu"
                        elif self.game_state == "complete":
                            self.game_state = "menu"
                    
                    elif self.game_state == "select_mode":
                        if event.key == pygame.K_UP:
                            self.selected_option = max(0, self.selected_option - 1)
                        elif event.key == pygame.K_DOWN:
                            self.selected_option = min(1, self.selected_option + 1)
            
            if self.game_state == "menu":
                self.draw_menu()
            elif self.game_state == "select_mode":
                self.draw_mode_selection()
            elif self.game_state == "complete":
                self.draw_complete()
            
            pygame.display.flip()
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = PoseMatchGame()
    game.run()