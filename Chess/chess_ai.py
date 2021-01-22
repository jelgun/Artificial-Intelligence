import chess
import random
from numpy import inf

# constant values for evaluation func.
piece_values = {
	'p':100,
	'n':320,
	'b':330,
	'r':500,
	'q':900,
	'k':0
}
table_values = {}
table_values['p'] = [
	[0,  0,  0,  0,  0,  0,  0,  0],
	[50, 50, 50, 50, 50, 50, 50, 50],
	[10, 10, 20, 30, 30, 20, 10, 10],
	[5,  5, 10, 25, 25, 10,  5,  5],
	[0,  0,  0, 20, 20,  0,  0,  0],
	[5, -5,-10,  0,  0,-10, -5,  5],
	[5, 10, 10,-20,-20, 10, 10,  5],
	[0,  0,  0,  0,  0,  0,  0,  0]
]

table_values['n'] = [
	[-50,-40,-30,-30,-30,-30,-40,-50],
	[-40,-20,  0,  0,  0,  0,-20,-40],
	[-30,  0, 10, 15, 15, 10,  0,-30],
	[-30,  5, 15, 20, 20, 15,  5,-30],
	[-30,  0, 15, 20, 20, 15,  0,-30],
	[-30,  5, 10, 15, 15, 10,  5,-30],
	[-40,-20,  0,  5,  5,  0,-20,-40],
	[-50,-40,-30,-30,-30,-30,-40,-50]
]

table_values['b'] = [
	[-20,-10,-10,-10,-10,-10,-10,-20],
	[-10,  0,  0,  0,  0,  0,  0,-10],
	[-10,  0,  5, 10, 10,  5,  0,-10],
	[-10,  5,  5, 10, 10,  5,  5,-10],
	[-10,  0, 10, 10, 10, 10,  0,-10],
	[-10, 10, 10, 10, 10, 10, 10,-10],
	[-10,  5,  0,  0,  0,  0,  5,-10],
	[-20,-10,-10,-10,-10,-10,-10,-20]
]

table_values['r'] = [
	[0,  0,  0,  0,  0,  0,  0,  0],
	[5, 10, 10, 10, 10, 10, 10,  5],
	[-5,  0,  0,  0,  0,  0,  0, -5],
	[-5,  0,  0,  0,  0,  0,  0, -5],
	[-5,  0,  0,  0,  0,  0,  0, -5],
	[-5,  0,  0,  0,  0,  0,  0, -5],
	[-5,  0,  0,  0,  0,  0,  0, -5],
	[0,  0,  0,  5,  5,  0,  0,  0]
]

table_values['q'] = [
	[-20,-10,-10, -5, -5,-10,-10,-20],
	[-10,  0,  0,  0,  0,  0,  0,-10],
	[-10,  0,  5,  5,  5,  5,  0,-10],
	[-5,  0,  5,  5,  5,  5,  0, -5],
	[0,  0,  5,  5,  5,  5,  0, -5],
	[-10,  5,  5,  5,  5,  5,  0,-10],
	[-10,  0,  5,  0,  0,  0,  0,-10],
	[-20,-10,-10, -5, -5,-10,-10,-20]
]

table_values['k'] = [
	[-30,-40,-40,-50,-50,-40,-40,-30],
	[-30,-40,-40,-50,-50,-40,-40,-30],
	[-30,-40,-40,-50,-50,-40,-40,-30],
	[-30,-40,-40,-50,-50,-40,-40,-30],
	[-20,-30,-30,-40,-40,-30,-30,-20],
	[-10,-20,-20,-20,-20,-20,-20,-10],
	[20, 20,  0,  0,  0,  0, 20, 20],
	[20, 30, 10,  0,  0, 10, 30, 20]
]


def evaluate(board, bot_color):
	white_score = 0
	black_score = 0

	fen = board.fen()
	# i and j represent a position on the board
	i = 0
	j = 0

	# parse the fen and calculate scores for each figure
	for ch in fen:
		if ch == '/':
			continue

		if ch.isdigit():
			j += int(ch)
		elif ch.isupper():
			white_score += piece_values[ch.lower()]
			white_score += table_values[ch.lower()][i][j]
			j += 1
		else:
			black_score += piece_values[ch]
			# 7-i and 7-j is written because table values are designed for white figures
			black_score += table_values[ch][7-i][7-j]
			j += 1

		if j == 8:
			i += 1
			j = 0

		if i == 8:
			break

	return (white_score - black_score) if bot_color else (black_score - white_score)


def minimax(board, bot_color, depth, player):
	# check ending conditions
	if board.is_checkmate():
		# player: 0 bot, 1 user
		if player == 1:
			# 10000 is checkmate score, other term is for finding shortest path to checkmate
			return 10000 - depth * 1000, None
		else:
			return -10000 + depth * 1000, None
	elif board.is_stalemate():
		return 0, None
	elif depth == 3:
		return evaluate(board, bot_color), None

	max_score = -inf
	min_score = inf
	optimal_play = None

	for play in board.legal_moves:
		# apply the move
		move = chess.Move.from_uci(str(play))
		board.push(move)

		# apply minimax to children
		score, _ = minimax(board, bot_color, depth + 1, 1 - player)

		# update score of current node
		if player == 0:
			if score > max_score:
				max_score = score
				optimal_play = str(play)
		else:
			if score < min_score:
				min_score = score
				optimal_play = str(play)

		# remove the applied move to reset the board for next moves
		board.pop()

	if player == 0:
		return max_score, optimal_play

	return min_score, optimal_play


def ai_play(board):
	bot_color = (board.fen()[-9] == 'w')
	_, optimal_play = minimax(board, bot_color, 0, 0)
	
	return optimal_play
