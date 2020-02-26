# Balanced datasets

# Digital Music
#mkdir results/single_task/digital_music
#mkdir results/single_task/digital_music/balanced
#touch results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv > results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --vocab_percent=0.5 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --vocab_percent=0.8 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --max_sent_len=100 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --max_sent_len=500 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --max_sent_len=1500 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --batch_size=16 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --batch_size=64 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --batch_size=128 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --batch_size=512 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --embedding_dim=10 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --embedding_dim=128 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --embedding_dim=300 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --dropout=0.3 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --dropout=0.7 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --pooling_layer=max >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --optimizer=sgd >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --optimizer=rmsprop >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --dense_units=5 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --dense_units=50 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --dense_units=100 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --num_dense_layers=2 >> results/single_task/digital_music/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Digital_Music_balanced_10.csv --num_dense_layers=3 >> results/single_task/digital_music/balanced/mlp.txt

# Pet Supplies
#mkdir results/single_task/pet_supplies
#mkdir results/single_task/pet_supplies/balanced
#touch results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv > results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --vocab_percent=0.5 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --vocab_percent=0.8 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --max_sent_len=100 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --max_sent_len=500 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --max_sent_len=1500 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --batch_size=16 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --batch_size=64 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --batch_size=128 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --batch_size=512 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --embedding_dim=10 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --embedding_dim=128 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --embedding_dim=300 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --dropout=0.3 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --dropout=0.7 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --pooling_layer=max >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --optimizer=sgd >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --optimizer=rmsprop >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --dense_units=5 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --dense_units=50 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --dense_units=100 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --num_dense_layers=2 >> results/single_task/pet_supplies/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Pet_Supplies_balanced_10.csv --num_dense_layers=3 >> results/single_task/pet_supplies/balanced/mlp.txt

# Beauty
#mkdir results/single_task/beauty
#mkdir results/single_task/beauty/balanced
#touch results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv > results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --vocab_percent=0.5 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --vocab_percent=0.8 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --max_sent_len=100 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --max_sent_len=500 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --max_sent_len=1500 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --batch_size=16 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --batch_size=64 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --batch_size=128 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --batch_size=512 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --embedding_dim=10 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --embedding_dim=128 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --embedding_dim=300 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --dropout=0.3 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --dropout=0.7 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --pooling_layer=max >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --optimizer=sgd >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --optimizer=rmsprop >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --dense_units=5 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --dense_units=50 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --dense_units=100 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --num_dense_layers=2 >> results/single_task/beauty/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Beauty_balanced_10.csv --num_dense_layers=3 >> results/single_task/beauty/balanced/mlp.txt
#
# Cell Phones
#mkdir results/single_task/cell_phones
#mkdir results/single_task/cell_phones/balanced
#touch results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv > results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --vocab_percent=0.5 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --vocab_percent=0.8 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --max_sent_len=100 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --max_sent_len=500 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --max_sent_len=1500 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --batch_size=16 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --batch_size=64 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --batch_size=128 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --batch_size=512 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --embedding_dim=10 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --embedding_dim=128 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --embedding_dim=300 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --dropout=0.3 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --dropout=0.7 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --pooling_layer=max >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --optimizer=sgd >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --optimizer=rmsprop >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --dense_units=5 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --dense_units=50 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --dense_units=100 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --num_dense_layers=2 >> results/single_task/cell_phones/balanced/mlp.txt
#time python3 single_task_nn.py --data=datasets/Cell_Phones_balanced_10.csv --num_dense_layers=3 >> results/single_task/cell_phones/balanced/mlp.txt

# Video Games
mkdir results/single_task/video_games
mkdir results/single_task/video_games/balanced
touch results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv > results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --vocab_percent=0.5 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --vocab_percent=0.8 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --max_sent_len=100 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --max_sent_len=500 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --max_sent_len=1500 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --batch_size=16 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --batch_size=64 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --batch_size=128 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --batch_size=512 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --embedding_dim=10 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --embedding_dim=128 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --embedding_dim=300 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --dropout=0.3 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --dropout=0.7 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --pooling_layer=max >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --optimizer=sgd >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --optimizer=rmsprop >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --dense_units=5 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --dense_units=50 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --dense_units=100 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --num_dense_layers=2 >> results/single_task/video_games/balanced/mlp.txt
time python3 single_task_nn.py --data=datasets/Video_Games_balanced_10.csv --num_dense_layers=3 >> results/single_task/video_games/balanced/mlp.txt

# Multiple products
