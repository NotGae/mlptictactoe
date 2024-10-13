import torch
# x가 이기면 1 return 아니면 0 return
def won(board, t=-1.):
    line = torch.tensor([t, t, t]).to(DEVICE)

    # 행비교
    for row in board:
        if torch.equal(row, line):
            return 1
    
    # 열비교
    for col in range(3):
        column = torch.tensor([board[row][col] for row in range(3)], dtype=torch.float).to(DEVICE)
        if torch.equal(column, line):
            return 1
    
    # 대각선 비교
    main_diagonal = torch.tensor([board[0][0], board[1][1], board[2][2]], dtype=torch.float).to(DEVICE)
    anti_diagonal = torch.tensor([board[0][0], board[1][1], board[2][2]], dtype=torch.float).to(DEVICE)
    if torch.equal(main_diagonal, line):
        return 1
    if torch.equal(anti_diagonal, line):
        return 1
    
    return 0

model = torch.jit.load('./2103_jit.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
label={'x':-1, 'o':1, 'b': 0}
input_count = 0
correct_count = 0
model.eval()
with torch.no_grad():
  while True:
    x = list(input().split(','))
    
    # # x에 o. x. b외의 값이 들어오면 break
    if (x.count('o') + x.count('x') + x.count('b')) != 9: break
    # # 구분된 개수가 9개가 아니라면 break
    if len(x) != 9: break

    input_count += 1
    data = torch.tensor([label[name] for name in x], dtype=torch.float).to(DEVICE)
    out = model(data)
    
    pred = out.argmax(dim=0)
    true_label = won(data.reshape(3,3))

    correct_count = correct_count + 1 if pred == true_label else correct_count
    winner = 'x' if pred == 1 else 'o'
    true_winner = 'x' if true_label == 1 else 'o'
    print(f'winner = {winner} true_winner = {true_winner}')

if input_count > 0:
  accuracy = correct_count / input_count
  print('accuracy = ', accuracy)