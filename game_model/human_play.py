import os
import cv2
import torch
import argparse
import tkinter as tk
import torchvision.transforms as transforms

from PIL import Image
from game import State
from dual_network import DualNetwork
from mcts import pv_mcts_scores, pv_mcts_action
from self_play import first_player_value, write_data
from cnn_model import OX_Model_CNN

class GameUI(tk.Frame):
    def __init__(self, net, size, pv_eval_count, temperture, game_count = None, master=None):
        tk.Frame.__init__(self, master)
        
        # about UI
        self.master.title('Tic Tac Toe')
        self.size = size
        self.game_count = game_count
        self.current_count = 0

        self.net = net
        self.net.load_state_dict(torch.load('./model/best.pth')) 
        net.eval()

        self.pv_eval_count = pv_eval_count
        self.temperature  = temperture
        self.next_action = pv_mcts_action(self.net, pv_eval_count=self.pv_eval_count, temperature = 0.0)

        # set video variable
        self.image_model = OX_Model_CNN()
        self.image_model.load_state_dict(torch.load('./model/OX_class_model.pth'))
        self.image_model.eval()

        # camera_setting
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.grid_x_size = 640//3
        self.grid_y_size = 480//3

        #camera transforms
        self.transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        success, self.img = self.cap.read()
        self.predict_result = [[0 for _ in range(3)] for _ in range(6)]

        self.c = tk.Canvas(self, width=self.size, height=self.size, highlightthickness=0)
    
        self.button = tk.Button(self, text="Camera", command=self.camera_check)
        self.button.pack()
        
        self.button = tk.Button(self, text="Submit", command=self.turn_of_human)
        self.button.pack()

        self.history = []
        self.current_history = []
        self.state = State()

        self.c.pack()
        self.on_draw()
        
    def camera_check(self):
        while True:
            success, self.img = self.cap.read()
            for i in range(1,3):
                cv2.line(self.img, (self.grid_x_size * i, 0), (self.grid_x_size *i, 480), (255,255,255), 2)
                cv2.line(self.img, (0, self.grid_y_size * i), (640, self.grid_y_size * i), (255,255,255), 2)
            cv2.imshow("Result", self.img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cv2.destroyWindow("Result")
        # self.cap.release()
        return
        
    def read_video(self):
        for i in range(1,3):
            cv2.line(self.img, (self.grid_x_size * i, 0), (self.grid_x_size *i, 480), (255,255,255), 2)
            cv2.line(self.img, (0, self.grid_y_size * i), (640, self.grid_y_size * i), (255,255,255), 2)
        cv2.imshow("Result", self.img)
        
    def update_idx(self):
        changed_idx = 0
        for i in range(3):
            for j in range(3):
                start_x, start_y = self.grid_x_size*j, self.grid_y_size*i
                end_x, end_y = start_x + self.grid_x_size, start_y + self.grid_y_size
                
                grid_img = self.img[start_y:end_y, start_x:end_x]
                
                saved_img_path = os.path.join('./predict', f'target_img_{i}_{j}.png')
                cv2.imwrite(saved_img_path, grid_img)
            
                saved_img = Image.open(saved_img_path).convert('L')
                saved_img = self.transform(saved_img)
                saved_img = saved_img.unsqueeze(0)

                with torch.no_grad():
                    output = self.image_model(saved_img)
                    _, predicted = torch.max(output, 1)
                    predicted_label = predicted.item()
                
                self.predict_result[i][j] = predicted_label
        
        idx = 0
        for m in range(3):
            for n in range(3):
                idx+=1
                if self.predict_result[m][n] == 1:
                    if self.predict_result[m][n] != self.predict_result[m+3][n]:
                        changed_idx = (idx-1)
        
        for k in range(3):
            for l in range(3):
                self.predict_result[k+3][l] = self.predict_result[k][l]

        return changed_idx        
    
    def turn_of_human(self):
        #success, self.img = self.cap.read()
        #self.read_video()
        in_index = self.update_idx()
        action = in_index
        if not (action in self.state.legal_actions()):
            return
        try:
            self.state = self.state.next(action)
            self.on_draw()
            self.master.after(1, self.turn_of_ai)
        except Exception as e:
            print(f"Error in turn_of_human : {e}")

    def turn_of_ai(self):
        if self.state.is_done():
            self.reset_game()
            return
        
        scores = pv_mcts_scores(self.net, self.state, self.pv_eval_count, self.temperature)

        # current history
        policies = [0] * 9
        for action, policy in zip(self.state.legal_actions(), scores):
            policies[action] = policy
        self.current_history.append([[self.state.pieces, self.state.enemy_pieces], policies, None])

        action = self.next_action(self.state)
        self.state = self.state.next(action)
        self.on_draw()

        if self.state.is_done():
            self.reset_game()

    def reset_game(self):
        # update history
        value = first_player_value(self.state)
        for i in range(len(self.current_history)):
            self.current_history[i][2] = value
            value = -value
        self.history.extend(self.current_history)
        #print(self.history)

        # initialize state & current history
        self.current_history = []
        self.state = State()
        self.on_draw()
        self.predict_result = [[0 for _ in range(3)] for _ in range(6)]
        #self.read_video()

        # check game_count & write history
        if self.game_count is not None:
            self.current_count += 1

            if self.current_count >= self.game_count:
                write_data(self.history)
                self.master.destroy()
    
    def draw_piece(self, idx, is_first_player):
        x = (idx % 3) * 80 + 10
        y = int(idx / 3) * 80 + 10

        if is_first_player:
            self.c.create_oval(x, y, x + 60, y + 60, width=2.0, outline='#FFFFFF') # O
        else:
            # X
            self.c.create_line(x, y, x + 60, y + 60, width=2.0, fill='#5D5D5D')
            self.c.create_line(x + 60, y, x, y + 60, width=2.0, fill='#5D5D5D')

    def on_draw(self):
        # draw board
        self.c.delete('all')
        self.c.create_rectangle(0, 0, self.size, self.size, width=0.0, fill='#01DF01')
        self.c.create_line(self.size/3, 0, self.size/3, self.size, width=2.0, fill='#000000')
        self.c.create_line(self.size/3*2, 0, self.size/3*2, self.size, width=2.0, fill='#000000')
        self.c.create_line(0, self.size/3, self.size, self.size/3, width=2.0, fill='#000000')
        self.c.create_line(0, self.size/3*2, self.size, self.size/3*2, width=2.0, fill='#000000')

        # draw piece
        for i in range(9):
            if self.state.pieces[i] == 1:
                self.draw_piece(i, self.state.is_first_player())
            if self.state.enemy_pieces[i] == 1:
                self.draw_piece(i, not self.state.is_first_player())
    
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

parser = argparse.ArgumentParser('Game UI')
parser.add_argument('--game_count', type=int, default=None)
parser.add_argument('--size', type=int, default=240)
parser.add_argument('--num_residual_block', type=int, default=16)
parser.add_argument('--num_filters', type=int, default=128)
parser.add_argument('--pv_eval_count', type=int, default=50)
parser.add_argument('--temperature', type=float, default=1.0)

if __name__ == '__main__':
    args = parser.parse_args()
    cap = cv2.VideoCapture(0)
    cap.release()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = DualNetwork(num_residual_block=args.num_residual_block, num_filters=args.num_filters).to(device)
    
    if args.game_count is not None:
        f = GameUI(net, args.size, args.pv_eval_count, args.temperature, game_count=args.game_count)
    else:
        f = GameUI(net, args.size, args.pv_eval_count, args.temperature)
    
    f.pack()
    f.mainloop()
