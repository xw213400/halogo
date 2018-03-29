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
    this.size = 7;
    this.user = 'b'; //'b': black, 'w': white, 'v': viewer(eve mode)
    this.komi = 6.5;
    this.next = 'b';
    this.bstart = false;
    this.uiBoard = Halo.Config.createWidget('board');
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
}

Object.assign(Board.prototype, {
    clear: function () {
        while (this.uiBoard.children.length > 0) {
            this.uiBoard.remove(this.uiBoard.children[0]);
        }

        var size = this.size;
        for (let i = 0; i !== size; ++i) {
            let uiLine = Halo.Config.createWidget('line');
            uiLine.formulaOffsetXFunc = function (w, h) {
                return w / size * (i + 0.475);
            };
            uiLine.formulaOffsetYFunc = function (w, h) {
                return w / size * 0.475;
            };
            uiLine.formulaSizeXFunc = function (w, h) {
                return w / size * 0.05;
            };
            uiLine.formulaSizeYFunc = function (w, h) {
                return w / size * (size - 0.95);
            };
            this.uiBoard.add(uiLine);

            uiLine = Halo.Config.createWidget('line');
            uiLine.formulaOffsetXFunc = function (w, h) {
                return w / size * 0.475;
            };
            uiLine.formulaOffsetYFunc = function (w, h) {
                return w / size * (i + 0.475);
            };
            uiLine.formulaSizeXFunc = function (w, h) {
                return w / size * (size - 0.95);
            };
            uiLine.formulaSizeYFunc = function (w, h) {
                return w / size * 0.05;
            };
            this.uiBoard.add(uiLine);
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
        var size = this.size;
        this.uiLastStone = this.next === 'b' ? Halo.Config.createWidget('black') : Halo.Config.createWidget('white');
        this.uiLastStone.setAnchorType(Halo.ANCHOR_LEFT, Halo.ANCHOR_BOTTOM);
        this.uiLastStone.formulaSizeXFunc = function (w, h) {
            return w / size * 0.9;
        };
        this.uiLastStone.formulaSizeYFunc = function (w, h) {
            return w / size * 0.9;
        };
        this.uiLastStone.formulaOffsetXFunc = function (w, h) {
            return w / size * (i + 0.05);
        };
        this.uiLastStone.formulaOffsetYFunc = function (w, h) {
            return w / size * (j + 0.05);
        };
        this.uiLastStone.alpha = 0.5;
        this.uiBoard.add(this.uiLastStone);

        this.stones[i + '_' + j] = this.uiLastStone;

        this.next = this.next === 'b' ? 'w' : 'b';
    }
});