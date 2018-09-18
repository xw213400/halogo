var Signals = {
    start: new Signal(),
    finish: new Signal(),
    passtip: new Signal()
};

var board = null;

function change_player_color() {
    if (board.user === 'b') {
        board.user = 'w';
        btnPlayerColor.setSprite('go_map.png', 'white_png');
    } else if (board.user === 'w') {
        board.user = 'b';
        btnPlayerColor.setSprite('go_map.png', 'black_png');
    }
}

function pass() {
    if (board.next === board.user) {
        board.execute(["play", board.user, 'pass']);
    }
}

function choose_black() {
    board.execute(["boardsize", board.size]);
}

function main_play() {
    if (board === null) {
        board = new Board();
    }

    function start(c) {
        var ui = Halo.Config.scene().getWidgetRoot().getChild('main');
        var prepare = ui.getChild('prepare');
        var pass = ui.getChild('pass');
        
        prepare.visible = false;
        pass.visible = true;

        board.user = c;
        board.execute(["boardsize", board.size]);
    }

    var flag = this.parent.flag;

    switch (flag) {
        case 'p':
        break;
        case 'b':
        case 'w':
            start(flag);
        break;
        default:
        break;
    }
}


var main = (function () {
    var funcs = Halo.ResourceManager.funcs();

    funcs["main_play"] = main_play;
}());