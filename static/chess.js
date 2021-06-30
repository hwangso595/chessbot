
// promotions
// castles
// en passant
// make api

let game = new Board();
let gameElement = document.getElementById('board');
let boardRect = gameElement.getBoundingClientRect();
let boardSize = boardRect.bottom - boardRect.top;
let interPieceDistancePixel = boardSize/WIDTH;
let interPieceDistancePercent = 100;
let pieces = document.getElementsByClassName('piece')
const mapTurnClass = {
    true: 'white',
    false: 'black'
}

gameElement.onload = resetBoard(true)

for (let piece of pieces) {
    piece.addEventListener('mousedown', handleMouseDownPiece)
}
for (let promotionPiece of document.getElementsByClassName('promotion-piece')) {
    promotionPiece.addEventListener('click', handleClickPromotion)
}
document.getElementById('promotions-container').addEventListener('click', handleClickPromotionCancel)

function handleMouseDownPiece(event) {
    event.preventDefault();
    console.log('mousedown')
    recheckDimension()
    game.move.piece = event.target.classList[0]
    if (game.turn == (game.move.piece == game.move.piece.toUpperCase())) {

        let posX = event.clientX - boardRect.left
        let posY = event.clientY - boardRect.top
        game.move.from = mapPosSquare(posX, posY, interPieceDistancePixel, game.flip)
        displayHelpers()
        
        console.log(game.move.from)
        event.target.classList.add('moving')
        document.addEventListener('mousemove', handleMouseMoveDocument)
        document.addEventListener('mouseup', handleMouseUpDocument)
    }
}

function handleMouseMoveDocument(event) {
    recheckDimension()
    piece = document.querySelector('.moving')
    piece.style.transform = `translateX(${event.clientX - boardRect.left - interPieceDistancePixel/2}px) translateY(${event.clientY - boardRect.top - interPieceDistancePixel/2}px)`;
}

function handleMouseUpDocument(event) {
    console.log('mouseup')
    recheckDimension()
    clearHelpers()
    document.removeEventListener('mousemove', handleMouseMoveDocument)
    piece = document.querySelector('.moving')

    let posX = event.clientX - boardRect.left
    let posY = event.clientY - boardRect.top
    game.move.to = mapPosSquare(posX, posY, interPieceDistancePixel, game.flip)

    if (isCancelMove()) {
        cancelMove(piece)
        piece.classList.remove('moving')
    } else {
        let toRank = Math.floor(game.move.to/LENGTH)
        if (game.move.piece.toLowerCase() === 'p' && (toRank === 7 || toRank === 0)) {
            let [newPosX, newPosY] = mapSquarePos(game.move.to, interPieceDistancePercent, game.flip)
            piece.style.transform = `translateX(${newPosX}%) translateY(${newPosY}%)`;
            let top = game.flip ? toRank === 0: toRank === 7;
            showPromotions(top, piece)
        } else {
            let info = game.makeMove()
            displayMove(piece, info)
            piece.classList.remove('moving')
            sendMoveRequest(info['uci'])
        }
    }

    document.removeEventListener('mouseup', handleMouseUpDocument)
}

function displayMove(piece, {moveTo, type, captureSquare, castleRookFrom, castleRookTo, uci}) {
    console.log({moveTo, type, captureSquare, castleRookFrom, castleRookTo, uci})
    if (type === 'cp' || type === 'ep') {
        let [captureX, captureY] = mapSquarePos(captureSquare, interPieceDistancePercent, game.flip)
        let captureClass = mapTurnClass[game.turn]

        let capturePiece = document.querySelector(`.piece.${captureClass}[style*='translateX(${captureX}%) translateY(${captureY}%)']`)
        if (capturePiece !== null) {
            capturePiece.remove()
        }
    }
    if (type === 'cs') {
        let [rookX, rookY] = mapSquarePos(castleRookFrom, interPieceDistancePercent, game.flip)
        let [toX, toY] = mapSquarePos(castleRookTo, interPieceDistancePercent, game.flip)
        let rook = document.querySelector(`.piece[style*='translateX(${rookX}%) translateY(${rookY}%)']`)
        console.log(rook)
        if (rook !== null) {
            rook.style.transform = `translateX(${toX}%) translateY(${toY}%)`
        }
    }
    let [newPosX, newPosY] = mapSquarePos(moveTo, interPieceDistancePercent, game.flip)
    piece.style.transform = `translateX(${newPosX}%) translateY(${newPosY}%)`;
}

async function testAjax() {
    try {
        let request = new Request('/test?turn=hello', {method: 'GET'});
        let response = await fetch(request)
        let {test} = await response.json()
        console.log(test)
    } catch (err) {
        console.log('ERROR: ', err)
    }
}

function displayHelpers() {
    let helperContainer = document.getElementById('helpers-container')
    if (game.helpers.hasOwnProperty(game.move.from)) {
        for (let square of game.helpers[game.move.from]) {
            let helper = document.createElement('div')
            helper.className = 'legal'
            let [newPosX, newPosY] = mapSquarePos(square, interPieceDistancePercent, game.flip)
            helper.style.transform = `translateX(${newPosX}%) translateY(${newPosY}%)`;
            helperContainer.appendChild(helper)
        }
    }
}

function clearHelpers() {
    let helperContainer = document.getElementById('helpers-container')
    helperContainer.innerHTML = ''
}

function displayWinner(winner) {
    let resultsContainer = document.getElementById('results')
    let result = winner === 'tie' ? 'TIE' : (winner === 'white' ? 'WHITE WON' : 'BLACK WON')
    resultsContainer.innerHTML = result
}

function makeBotMove({move, legal_uci, winner}) {
    if (winner !== '') {
        displayWinner(winner)
    } else {
        game.setMoveUci(move)
        let [fromX, fromY] = mapSquarePos(game.move.from, interPieceDistancePercent, game.flip)
        let piece = document.querySelector(`.piece[style*='translateX(${fromX}%) translateY(${fromY}%)']`)
        let info = game.makeMove()
        displayMove(piece, info)

        game.legalMoves = legal_uci
        game.createHelpers()
    }
}

async function startGame(isFirst) {
    try {
        let request = new Request(`/newgame?turn=${isFirst}`, {method: 'GET'});
        let response = await fetch(request)
        let {move, legal_uci, winner} = await response.json()
        console.log({move, legal_uci, winner, isFirst})
        if (!isFirst) {
            makeBotMove({move, legal_uci, winner})
        } else {
            game.legalMoves = legal_uci
            game.createHelpers()
        }
    } catch (err) {
        console.log('ERROR: ', err)
    }

}

async function sendMoveRequest(moveUci) {
    try {
        let request = new Request(`/makemove?uci=${moveUci}`, {method: 'GET'});
        let response = await fetch(request)
        let {move, legal_uci, winner} = await response.json()
        console.log({move, legal_uci, winner})
        makeBotMove({move, legal_uci, winner})
    } catch (err) {
        console.log('ERROR: ', err)
    }
}

function handleClickPromotion (event) {
    event.stopPropagation();
    event.cancelBubble = true;
    pieceType = event.target.classList[0]
    game.move.promotion = pieceType
    let [posX, posY] = mapSquarePos(game.move.to, interPieceDistancePercent, game.flip)
    let originTurn = mapTurnClass[game.turn]
    let piece = document.querySelector('.moving')
    piece.classList.replace(piece.classList[0], pieceType)
    let info = game.makeMove()
    displayMove(piece, info)
    clearPromotionView();
    piece.classList.remove('moving')
    sendMoveRequest(info['uci'])
}

function handleClickPromotionCancel (event) {
    let piece = document.querySelector(`.moving`)
    cancelMove(piece)
    clearPromotionView();
    piece.classList.remove('moving')
}

function clearPromotionView() {
    promotionContainer = document.getElementById('promotions-container')
    for (let node of promotionContainer.children) {
        node.style.display = 'none'
    }
    promotionContainer.style.display = 'none'
}

function showPromotions(top, piece) {
    // show promotion squares at correct position
    if (top) {
        promotions = document.querySelector('.top')
    } else {
        promotions = document.querySelector('.bottom')
    }
    
    promotions.style.transform = piece.style.transform.split(' ')[0]
    promotions.parentElement.style.display = 'block'
    promotions.style.display = 'block'
}

function isCancelMove() {
    if (!game.helpers.hasOwnProperty(game.move.from)) {
        return true;
    }
    return !game.helpers[game.move.from].includes(game.move.to)
}

function cancelMove(piece) {
    let [originX, originY] = mapSquarePos(game.move.from, interPieceDistancePercent, game.flip)
    piece.style.transform = `translateX(${originX}%) translateY(${originY}%)`;
    game.clearMove()
}

function recheckDimension() {
    gameElement = document.getElementById('board');
    boardRect = gameElement.getBoundingClientRect();
    boardSize = boardRect.bottom - boardRect.top;
    interPieceDistancePixel = boardSize/WIDTH;
}

function resetBoard(isFirst) {
    let resultsContainer = document.getElementById('results')
    resultsContainer.innerHTML = ''
    recheckDimension()
    clearPromotionView()
    console.log('resetting board')
    if (isFirst == game.flip) {
        flipBoard()
    }
    game.resetGame()
    displayState()
    pieces = document.getElementsByClassName('piece')
    for (let piece of pieces) {
        piece.addEventListener('mousedown', handleMouseDownPiece)
    }
    startGame(isFirst)
}

function flipBoard() {
    let topPromotion = document.querySelector('.top')
    let bottomPromotion = document.querySelector('.bottom')
    topPromotion.classList.replace('top', 'bottom')
    bottomPromotion.classList.replace('bottom', 'top')
    recheckDimension()
    console.log('flip board')
    game.flip = !game.flip
    displayState()
    pieces = document.getElementsByClassName('piece')
    for (let piece of pieces) {
        piece.addEventListener('mousedown', handleMouseDownPiece)
    }
}

function displayState() {
    gameElement.innerHTML = ''
    for (let i = 0; i < game.state.length; i++) {
        if (game.state[i] !== '') {
            let piece = document.createElement('div');
            let isWhite = game.state[i].toUpperCase() === game.state[i]
            piece.className = game.state[i] + ' ' + 'piece' + ' ' + mapTurnClass[isWhite];
            let [x, y] = mapSquarePos(i, interPieceDistancePercent, game.flip);
            piece.style.transform = `translateX(${x}%) translateY(${y}%)`;
            gameElement.appendChild(piece)
        }
    }
}

function printBoard() {
    for (let i = 0; i < WIDTH; i++) {
        let row = [String(WIDTH - i), ' ']
        for (let j = 0; j < WIDTH; j++) {
            if (game.state[(WIDTH - i - 1) * WIDTH + j] !== '') {
                row.push(game.state[(WIDTH - i - 1) * WIDTH + j])
            } else {
                row.push('.')
            }
                
        }
        console.log(row.join(' '))
    }
}
