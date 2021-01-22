############################################
#             Careful Student!             #
# There is nothing to change here for you  #
# The script is carefully coded by experts #
############################################

from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QFrame
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QSize
from PyQt5 import *
from PyQt5.QtCore import *
import chess
import os
import time
from chess_ai import ai_play

#piece urls for visual
piece_urls = {"r":"br.png", "b":"bb.png", "n":"bn.png", "k":"bk.png", "q":"bq.png", "p":"bp.png",
		"R":"wr.png", "B":"wb.png", "N":"wn.png", "K":"wk.png", "Q":"wq.png", "P":"wp.png", "1":"blank"}

#returns corresponding url (depending of type of piece) for that square
def get_piece(full_fen, index):
	my_fen = ""
	fen = full_fen.split(" ")[0]

	for char in fen:

		if char.isdigit():
			for i in range(int(char)):
				my_fen += "1"
		else:
			my_fen += char

	row = my_fen.split("/")[index//8]
	return  os.path.join(os.path.join(os.getcwd(), "icons"), piece_urls[row[index%8]])

#The buttons are numbered from 0-63, so we need to convert "8" into "h8" to represent moves
def from_position_to_chess_move(position):
	move = ""

	if position % 8 == 0:
		move += "a"
	elif position % 8 == 1:
		move += "b"
	elif position % 8 == 2:
		move += "c"
	elif position % 8 == 3:
		move += "d"
	elif position % 8 == 4:
		move += "e"
	elif position % 8 == 5:
		move += "f"
	elif position % 8 == 6:
		move += "g"
	elif position % 8 == 7:
		move += "h"

	move += str(8 - (position // 8))

	return move

#After each move, board is updated. It is inefficient to update them all but I don't have much time.
def update_board(board, buttons):
	for i in range(8):
		for j in range(8):
			if get_piece(board.fen(), (i * 8) + j) != "blank":
				buttons[i*8+j].setIcon(QIcon(get_piece(board.fen(), (i * 8) + j)))
			else:
				buttons[i*8+j].setIcon(QIcon())

#Disconnect the click event, so that no move can occur after game over.
def game_over(buttons):
	for button in buttons:
		button.disconnect()

class App(QWidget):

	def __init__(self, board):
		super().__init__()
		self.title = 'CENG461 Chess'
		self.left = 10
		self.top = 10
		self.width = 500
		self.height = 500
		self.board = board
		self.first_position = "none"
		self.second_position = "none"
		self.buttons = []
		self.initUI()

	def initUI(self):
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

		self.label = QLabel(self)
		self.label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
		if self.board.fen().split(" ")[1] == "w": 
			self.label.setText("Turn: White")
		else:
			self.label.setText("Turn: Black")
		self.label.setAlignment(Qt.AlignBottom | Qt.AlignLeft)

		for i in range(8):
			for j in range(8):
				button = QPushButton('', self)
				button.move((j+1)*50, (i+1)*50)
				button.setObjectName('Button ' + str((i*8)+j))

				if get_piece(self.board.fen(), (i*8)+j) != "blank":
					button.setIcon(QIcon(get_piece(self.board.fen(), (i*8)+j)))

				button.setIconSize(QSize(50, 50))
				button.resize(50,50)
				button.clicked.connect(self.on_click)

				if ((i%2)+j)%2 == 0: #clever way to change black to white
					button.setStyleSheet("background-color: #835C3B")
				else:
					button.setStyleSheet("background-color: white")

				self.buttons.append(button)
		
		#codes below (except self.show()) is only for making AI to play first move
		ai_move = ai_play(self.board.copy()) 
		

		if self.board.is_legal(chess.Move.from_uci(ai_move)):
			self.board.push(chess.Move.from_uci(ai_move))
		else:
			raise ValueError('Your AI Made Illegal Move')

		update_board(self.board, self.buttons)
		if self.board.is_checkmate():
			self.label.setText("CheckMate") #Indicates someone win by checkmate
			game_over(self.buttons)
		elif self.board.is_game_over():
			self.label.setText("Game Over") #Indicates that game is over due to non-checkmate scenario (maybe draw)
			game_over(self.buttons)
		else:
			if self.board.fen().split(" ")[1] == "w": 
				self.label.setText("Turn: White")
			else:
				self.label.setText("Turn: Black")


		self.show()

	@pyqtSlot()
	def on_click(self):
		position = int(self.sender().objectName().split(" ")[1]) #finding the position of the button

		if self.first_position == "none":
			self.first_position = position
			self.buttons[position].setStyleSheet("background-color: blue")

		else:
			self.second_position = position

			if from_position_to_chess_move(self.first_position) != from_position_to_chess_move(self.second_position):

				algebraic_move = from_position_to_chess_move(self.first_position) + \
								 from_position_to_chess_move(self.second_position)

				# Here I also check promotion of pawn by hand. This is silly check, makes code dirty. But saves time.
				if self.board.is_legal(chess.Move.from_uci(algebraic_move)) or self.board.is_legal(chess.Move.from_uci(algebraic_move+"q")):
					if self.board.is_legal(chess.Move.from_uci(algebraic_move+"q")): #You can only promote to queen but it is the best anyway :)
						algebraic_move += "q"

					self.board.push(chess.Move.from_uci(algebraic_move))
					update_board(self.board, self.buttons)

					if self.board.is_checkmate():
						self.label.setText("CheckMate")
						game_over(self.buttons)
						return 0

					elif self.board.is_game_over():
						self.label.setText("Game Over by non-checkmate") #Indicates that game is over due to non-checkmate scenario (maybe draw)
						game_over(self.buttons)
						return 0

					else:
						if self.board.fen().split(" ")[1] == "w": #somehow label couln't get updated in update_board
							self.label.setText("Turn: White")
						else:
							self.label.setText("Turn: Black")

					ai_move = ai_play(self.board.copy()) # get AI move, but we give AI copy of the board so it can play with it.

					if self.board.is_legal(chess.Move.from_uci(ai_move)):
						self.board.push(chess.Move.from_uci(ai_move))
					else:
						raise ValueError('Your AI Made Illegal Move')

					update_board(self.board, self.buttons)

					if self.board.is_checkmate():
						self.label.setText("CheckMate")
						game_over(self.buttons)
						return 0

					elif self.board.is_game_over():
						self.label.setText("Game Over by non-checkmate")
						game_over(self.buttons)
						return 0
					else:
						if self.board.fen().split(" ")[1] == "w": #somehow label couln't get updated in update_board
							self.label.setText("Turn: White")
						else:
							self.label.setText("Turn: Black")
					
			# The button in first_position had blue background because of selection, but now it needs to be reset.
			if (((self.first_position // 8) % 2) + (self.first_position % 8)) % 2 == 0:
				self.buttons[self.first_position].setStyleSheet("background-color: #835C3B")
			else:
				self.buttons[self.first_position].setStyleSheet("background-color: white")

			self.first_position = "none"
			self.second_position = "none"



