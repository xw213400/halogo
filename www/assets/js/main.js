var Signals = {
    start: new Signal(),
    finish: new Signal(),
    passtip: new Signal()
};

var scene = null;
var ui = null;
var uiMain = null;
var btnAddBoardSize = null;
var txtBoardSize = null;
var btnSubBoardSize = null;
var btnPlayerColor = null;
var txtKomi = null;
var btnPass = null;
var txtInfo = null;
var board = null;

function boardsize(s) {
    board.size += s;

    if (board.size >= 19) {
        btnAddBoardSize.setEnable(false);
    } else if (!btnAddBoardSize.enable) {
        btnAddBoardSize.setEnable(true);
    }
    if (board.size <= 7) {
        btnSubBoardSize.setEnable(false);
    } else if (!btnSubBoardSize.enable) {
        btnSubBoardSize.setEnable(true);
    }

    txtBoardSize.text = 'Board Size: ' + board.size;
    txtBoardSize.refresh();

    board.clear();
}

function komi(k) {
    board.komi += k;
    txtKomi.text = 'Komi: ' + board.komi;
    txtKomi.refresh();
}

function add_board_size() {
    boardsize(2);
}

function sub_board_size() {
    boardsize(-2);
}

function change_player_color() {
    if (board.user === 'b') {
        board.user = 'w';
        btnPlayerColor.setSprite('go_map.png', 'white_png');
    } else if (board.user === 'w') {
        board.user = 'b';
        btnPlayerColor.setSprite('go_map.png', 'black_png');
    }
}

function add_komi() {
    komi(0.5);
}

function sub_komi() {
    komi(-0.5);
}

function pass() {
    if (board.next === board.user) {
        board.execute(["play", board.user, 'pass']);
    }
}

function pve() {
    board.execute(["boardsize", board.size]);
}

function eve() {
    board.user = 'v';
    board.execute(["boardsize", board.size]);
}

function initBoard() {
    scene = Halo.Config.scene();
    ui = scene.getWidgetRoot();
    uiMain = Halo.Config.createWidget('main');
    btnAddBoardSize = uiMain.getChild('btn_add_board_size');
    txtBoardSize = uiMain.getChild('txt_board_size');
    btnSubBoardSize = uiMain.getChild('btn_sub_board_size');
    btnPlayerColor = uiMain.getChild('btn_player_color');
    txtKomi = uiMain.getChild('txt_komi');
    btnPass = Halo.Config.createWidget('pass');
    txtInfo = Halo.Config.createWidget('info');
    board = new Board();

    txtInfo.visible = false;
    ui.add(uiMain);
    ui.add(board.uiBoard);
    ui.add(txtInfo);
    ui.relayout();
    board.clear();

    Signals.start.add(function () {
        ui.remove(uiMain);
        ui.add(btnPass);
    });

    Signals.finish.add(function (score) {
        ui.remove(btnPass);
        txtInfo.text = score;
        txtInfo.visible = true;
        txtInfo.refresh();
    });

    Signals.passtip.add(function () {
        txtInfo.visible = true;
        txtInfo.text = "PASS";
        txtInfo.refresh();
        setTimeout(function () {
            txtInfo.visible = false;
        }, 3000);
    });
}


var main = (function () {
    var funcs = Halo.ResourceManager.funcs();

    funcs["initBoard"] = initBoard;
    funcs["add_board_size"] = add_board_size;
    funcs["sub_board_size"] = sub_board_size;
    funcs["change_player_color"] = change_player_color;
    funcs["add_komi"] = add_komi;
    funcs["sub_komi"] = sub_komi;
    funcs["pass"] = pass;
    funcs["pve"] = pve;
    funcs["eve"] = eve;
}());

if (window.func_unloaders) {
    window.func_unloaders['main.js'] = function () {
        var funcs = Halo.ResourceManager.funcs();

        delete funcs["initBoard"];
        delete funcs["add_board_size"];
        delete funcs["sub_board_size"];
        delete funcs["change_player_color"];
        delete funcs["add_komi"];
        delete funcs["sub_komi"];
        delete funcs["pass"];
        delete funcs["pve"];
        delete funcs["eve"];
    }
}