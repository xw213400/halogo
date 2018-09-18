function gtp_move(i, j) {
    if (i >= 8) {
        i++;
    }

    return String.fromCharCode(65 + i, 49 + j);
}

function json_move(move) {
    var i = move.charCodeAt(0) - 65;
    if (i >= 9) { i--; }
    j = parseInt(move.substr(1)) - 1;
    return { i: i, j: j };
}

function Board() {
    this.size = 9;
    this.user = 'b'; //'b': black, 'w': white, 'v': viewer(eve mode)
    this.komi = 5.5;
    this.next = 'b';
    this.bstart = false;
    this.uiBoard = Halo.Config.scene().getWidgetRoot().getChild('board');
    this.uiLastStone = null;
    this.stones = {};
    this.pass = 0;
    this.finish = false;

    var board = this;
    this.uiBoard.addEventListener(Halo.EVT_UI_DOWN, function (evt) {
        evt.stop = true;
        if (board.bstart && board.next === board.user && !this.finish) {
            var x = evt.x - board.uiBoard.dimension.min.x;
            var y = evt.y - board.uiBoard.dimension.min.y;
            board.click(x, y);
        }
    });

    this.clear();
}

Object.assign(Board.prototype, {
    clear: function () {
        while (this.uiBoard.children.length > 0) {
            this.uiBoard.remove(this.uiBoard.children[0]);
        }

        var size = this.size;
        var unit = 'w/' + size + '*';
        for (let i = 0; i !== size; ++i) {
            let uiLine = Halo.Config.createWidget('line');
            uiLine.setFormulaOffsetX(unit + '('+ i + '+0.475)');
            uiLine.setFormulaOffsetY(unit + '0.475');
            uiLine.setFormulaSizeX(unit + '0.05');
            uiLine.setFormulaSizeY(unit + '(' + size + '-0.95)');
            this.uiBoard.add(uiLine);

            uiLine = Halo.Config.createWidget('line');
            uiLine.setFormulaOffsetX(unit + '0.475');
            uiLine.setFormulaOffsetY(unit + '('+ i + '+0.475)');
            uiLine.setFormulaSizeX(unit + '(' + size + '-0.95)');
            uiLine.setFormulaSizeY(unit + '0.05');
            this.uiBoard.add(uiLine);
        }
    },

    add: function (i, j, c) {
        var wt_stone = c === 'b' ? Halo.Config.createWidget('black') : Halo.Config.createWidget('white');

        this.stones[i + '_' + j] = wt_stone;

        var size = this.size;

        wt_stone.setAnchorType(Halo.ANCHOR_LEFT, Halo.ANCHOR_BOTTOM);

        var unit = 'w/' + size + '*';
        wt_stone.setFormulaSizeX(unit + '0.9');
        wt_stone.setFormulaSizeY(unit + '0.9');
        wt_stone.setFormulaOffsetX(unit + '(' + i + '+0.05)');
        wt_stone.setFormulaOffsetY(unit + '(' + j + '+0.05)');

        this.uiBoard.add(wt_stone);

        return wt_stone;
    },

    setBoard: function (bb) {
        for (var j = 0; j !== this.size; ++j) {
            for (var i = 0; i !== this.size; ++i) {
                this.remove(i, j);
                var c = bb[j * this.size + i];
                if (c !== 0) {
                    this.add(i, j, c === 1 ? 'b' : 'w');
                }
            }
        }
    },

    execute: function (cmd) {
        var scope = this;

        function check_caps(caps) {
            caps = caps.split(',');
            for (var i = 0; i !== caps.length; ++i) {
                var move = json_move(caps[i]);
                scope.remove(move.i, move.j);
            }
        }

        var request = new XMLHttpRequest();
        request.open('POST', '/execute', true);
        request.setRequestHeader('Content-Type', 'application/json');
        request.onreadystatechange = () => {
            if (request.readyState === 4) {
                console.log(request.response);
                if (request.status === 200) {
                    var msg = JSON.parse(request.response);
                    if (msg[0] !== '=') {
                        return;
                    }
                    switch (cmd[0]) {
                        case 'boardsize':
                            board.execute(["komi", this.komi]);
                            break;
                        case 'komi':
                            this.bstart = true;
                            Signals.start.dispatch();
                            if (this.user === 'v' || this.user !== this.next) {
                                this.execute(['genmove', this.next]);
                            }
                            break;
                        case 'play':
                            if (msg[1] === 'pass') {
                                this.passMove();
                            } else {
                                var move = json_move(msg[1]);
                                this.move(move.i, move.j);
                            }
                            if (msg.length >= 3) {
                                check_caps(msg[2]);
                            }
                            this.execute(['genmove', this.next]);
                            break;
                        case 'genmove':
                            if (msg[1] === 'pass') {
                                this.passMove();
                                if (this.pass >= 2) {
                                    this.finish = true;
                                    this.execute(['score']);
                                } else {
                                    Signals.passtip.dispatch();
                                }
                            } else {
                                var move = json_move(msg[1]);
                                this.move(move.i, move.j);
                            }
                            if (msg.length >= 3) {
                                check_caps(msg[2]);
                            }
                            if (this.user === 'v' && !this.finish) {
                                this.execute(['genmove', this.next]);
                            }
                            break;
                        case 'score':
                            Signals.finish.dispatch(msg[1]);
                            break;
                        default:
                            break;
                    }
                }
            }
        }
        request.send(JSON.stringify(cmd));
    },

    click: function (x, y) {
        var gridsize = this.uiBoard.size.x / this.size;
        var i = Math.floor(x / gridsize);
        var j = Math.floor(y / gridsize);
        var x = x % gridsize;
        var y = y % gridsize;
        var range = gridsize * 0.4;
        var center = gridsize * 0.5;

        if (Math.abs(x - center) < range && Math.abs(y - center) < range) {
            this.execute(['play', this.next, gtp_move(i, j)]);
        }
    },

    remove: function (i, j) {
        var key = i + '_' + j;
        var stone = this.stones[key];
        if (stone) {
            delete this.stones[key];
            this.uiBoard.remove(stone);
            if (stone === this.uiLastStone) {
                this.uiLastStone = null;
            }
        }
    },

    passMove: function () {
        this.pass++;
        if (this.uiLastStone) {
            this.uiLastStone.alpha = 1;
        }
        this.next = this.next === 'b' ? 'w' : 'b';
    },

    move: function (i, j) {
        this.pass = 0;
        if (this.uiLastStone) {
            this.uiLastStone.alpha = 1;
        }

        this.uiLastStone = this.add(i, j, this.next);
        this.uiLastStone.alpha = 0.5;

        this.stones[i + '_' + j] = this.uiLastStone;

        this.next = this.next === 'b' ? 'w' : 'b';
    }
});