const LENGTH = 8;
const WIDTH = 8;
const STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const turn = {
    WHITE: true,
    BLACK: false
}
const mapSqareNumAlg = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
const mapSqareAlgNum = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

function mapMoveUci({from, to, promotion}) {
    return mapSquareUci(from) + mapSquareUci(to) + promotion.toLowerCase()
}

function mapSquareUci(square) {
    return mapSqareNumAlg[square % WIDTH] + String(Math.floor(square/WIDTH) + 1)
}

function mapUciSquare(uci) {
    return mapSqareAlgNum[uci.charAt(0)] + LENGTH * (parseInt(uci.charAt(1)) - 1)
}

function mapSquarePos(idx, interPieceDistancePercent, bside) {
    if (bside) {
        return [(LENGTH - idx % WIDTH - 1) * interPieceDistancePercent, (Math.floor(idx / WIDTH)) * interPieceDistancePercent]
    } else {
        return [(idx % WIDTH) * interPieceDistancePercent, (WIDTH - Math.floor(idx / WIDTH) - 1) * interPieceDistancePercent]
    }
}

function mapPosSquare(pixelX, pixelY, interPieceDistancePixel, bside) {
    if (bside) {
        return (LENGTH - Math.floor(pixelX/interPieceDistancePixel) - 1) + WIDTH * Math.floor(pixelY/interPieceDistancePixel)
    } else {
        return Math.floor(pixelX/interPieceDistancePixel) + WIDTH * (WIDTH - Math.floor(pixelY/interPieceDistancePixel)- 1)
    }
}

class Board {
    constructor(fen=STARTING_FEN) {
        this.state = new Array(LENGTH * WIDTH).fill('');
        this.turn = turn.WHITE;
        // flip does not affect the state. Only the visual representation of the board
        this.flip = false;
        this.move = {
            from: -1,
            to: -1,
            piece: '',
            promotion: ''
        }
        this.setState(fen);
        this.legalMoves = []
        this.helpers = {}
    }

    makeMove() {
        // if (! this.isValidMove()) return {moveTo: this.move.from, type:'nv', captureSquare:-1, castleRookFrom:-1, castleRookTo:-1, uci:''}}
        let info = {
            moveTo: -1,
            type:'',
            captureSquare:-1,
            castleRookFrom:-1,
            castleRookTo:-1,
            uci: ''
        }
        let moveDiff = this.move.to - this.move.from
        this.state[this.move.from] = ''
        // en passant
        if (this.move.piece.toLowerCase() === 'p' && (Math.abs(moveDiff) === 7 || Math.abs(moveDiff) === 9) && this.state[this.move.to] === '') {
            if (moveDiff < 0) {
                info.captureSquare = this.move.to + 8
            } else if (moveDiff > 0) {
                info.captureSquare = this.move.to - 8
            }
            info.type = 'ep'
            this.state[info.captureSquare] = ''
        }

        // regular capture
        if (this.state[this.move.to] !== '') {
            info.type = 'cp'
            info.captureSquare = this.move.to
        }

        // promotion
        if (this.move.promotion !== '') {
            this.state[this.move.to] = this.move.promotion
        } else {
            this.state[this.move.to] = this.move.piece
        }

        // castling
        if (this.move.piece.toLowerCase() === 'k' && Math.abs(moveDiff) === 2) {
            if (moveDiff > 0) {
                info.castleRookFrom = this.move.from + 3
                info.castleRookTo = this.move.from + 1
            } else {
                info.castleRookFrom = this.move.from - 4
                info.castleRookTo = this.move.from - 1
            }
            this.state[info.castleRookTo] = this.state[info.castleRookFrom]
            this.state[info.castleRookFrom] = ''
            info.type = 'cs'
        }
        info.uci = mapMoveUci(this.move)
        
        console.log(this.move)
        info.moveTo = this.move.to
        printBoard()
        // update state
        // request the bot to make board
        // on response store available moves and new board fen
        // update state
        this.turn = !this.turn
        this.clearMove()
        // type can be '' (move), 'nv' (not valid), 'cp' (capture), 'cs' (castles), 'ep' (en pessant)
        // 'cp', 'ep' also returns a captureSquare else -1
        // 'cs' returns the rook from and to squares
        return info
    }

    setMoveUci(uci) {
        this.move.from = mapUciSquare(uci.substr(0,2))
        this.move.to = mapUciSquare(uci.substr(2,4))
        this.move.promotion = uci.charAt(4)
        this.move.piece = this.state[this.move.from]
    }

    isValidMove() {
        let uci = mapSquareUci(this.move.from) + moveSquareUci(this.move.to) + this.move.promotion
        return this.legalMoves.includes(uci)
    }

    createHelpers() {
        let moveFrom = -1
        let moveTo = -1
        this.helpers = {}
        for (let move of this.legalMoves) {
            moveFrom = mapUciSquare(move.substr(0,2))
            moveTo = mapUciSquare(move.substr(2,4))
            if (! this.helpers.hasOwnProperty(moveFrom)){
                this.helpers[moveFrom] = []
            }
            this.helpers[moveFrom].push(moveTo)
        }
    }
    
    setState(fen) {
        let rows = fen.split(' ')[0].split('/')
        this.state.fill('')
        for (let i = 0; i < WIDTH; i++) {
            for (let j = 0; j < LENGTH; j++) {
                if (isNaN(rows[i].charAt(j))) {
                    this.state[i * WIDTH + j] = rows[WIDTH - i - 1].charAt(j)
                }
            }
        }
    }

    resetGame() {
        this.setState(STARTING_FEN)
        this.turn = turn.WHITE;
        this.clearMove()
        this.legalMoves = []
        this.helpers = {}
    }

    clearMove() {
        this.move = {
            from: -1,
            to: -1,
            piece: '',
            promotion: ''
        }
    }
}
