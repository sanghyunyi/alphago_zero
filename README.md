# alphago_zero
A reproduction of Alphago Zero in "Mastering the game of Go without human knowledge" using Tensorflow 1.3.

This code is based on 'https://github.com/Rochester-NRT/RocAlphaGo'.

Now the board size is 5. (You can configure the size by changing board_size value.)

## train
Run train.sh 

## play
Playing with the best trained model.

python play_game.py

## TODO
Implement resignation threshold in self play part.
