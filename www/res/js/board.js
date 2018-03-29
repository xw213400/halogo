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

    execute: function (msg) {
        var request = new XMLHttpRequest();
        request.open('POST', 'http://localhost:8008/execute', true);
        request.setRequestHeader('Content-Type', 'application/json');
        request.onreadystatechange = () => {
            if (request.readyState === 4) {
                console.log(request.response);
                if (request.status === 200) {
                    var res = JSON.parse(request.response);
                    if (res.ok) {
                        if (res.cmd === 'move') {
                            if (res.pass) {
                                this.passMove();
                            } else {
                                this.move(res.i, res.j);
                            }
                            this.execute({ cmd: 'genmove', color: this.next });
                        } else if (res.cmd === 'start') {
                            this.bstart = true;
                            Signals.start.dispatch();
                            if (this.user === 'v' || this.user !== this.next) {
                                this.execute({ cmd: 'genmove', color: this.next });
                            }
                        } else if (res.cmd === 'genmove') {
                            if (res.pass) {
                                this.passMove();
                                if (this.pass >= 2) {
                                    this.finish = true;
                                    this.execute({ cmd: 'score' });
                                } else {
                                    Signals.genpass.dispatch();
                                }
                            } else {
                                this.move(res.i, res.j);
                            }
                            if (this.user === 'v' && !this.finish) {
                                this.execute({ cmd: 'genmove', color: this.next });
                            }
                        } else if (res.cmd === 'score') {
                            Signals.finish.dispatch(res.score);
                        }
                        if (res.caps) {
                            for (var i = 0; i !== res.caps.length; ++i) {
                                var cap = res.caps[i];
                                this.remove(cap.i, cap.j);
                            }
                        }
                    } else {
                        alert(res.msg);
                    }
                }
            }
        }
        request.send(JSON.stringify(msg));
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
            this.execute({
                cmd: 'move',
                color: this.next,
                i: i,
                j: j
            });
        }
    },

    remove: function(i, j) {
        var key = i+'_'+j;
        var stone = this.stones[key];
        if (stone) {
            delete this.stones[key];
            this.uiBoard.remove(stone);
            if (stone === this.uiLastStone) {
                this.uiLastStone = null;
            }
        }
    },

    passMove: function() {
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

        this.stones[i+'_'+j] = this.uiLastStone;

        this.next = this.next === 'b' ? 'w' : 'b';
    }
});