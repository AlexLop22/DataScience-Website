const boardElement = document.getElementById('board');
let currentPlayer = 'X';
const board = ['', '', '', '', '', '', '', '', ''];

function handleCellClick(index) {
    if (!board[index]) {
        board[index] = currentPlayer;
        const cell = document.createElement('div');
        cell.classList.add('cell');
        cell.innerText = currentPlayer;
        boardElement.appendChild(cell);
        currentPlayer = currentPlayer === 'X' ? 'O' : 'X';
    }
}

function resetBoard() {
    board.fill('');
    boardElement.innerHTML = '';
    currentPlayer = 'X';
}

for (let i = 0; i < 9; i++) {
    const cell = document.createElement('div');
    cell.classList.add('cell');
    cell.addEventListener('click', () => handleCellClick(i));
    boardElement.appendChild(cell);
}
