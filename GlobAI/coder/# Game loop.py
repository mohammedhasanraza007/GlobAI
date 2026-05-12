# Game loop
while True:
    # Player 1's turn
    if random.choice(['left', 'right']) == 'left':
        player1_score += 1
    else:
        player2_score += 1

    # Player 2's turn
    if random.choice(['left', 'right']) == 'left':
        player2_score += 1
    else:
        player1_score += 1

    # Check if either player has won
    if player1_score >= 5 or player2_score >= 5:
        print(f"Player {player1_score} wins!")
        break
    elif player1_score == player2_score:
        print("It's a tie!")
        break