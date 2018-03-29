var Signals = {
    board_size: new Signal(),
    player_color: new Signal(),
    komi: new Signal(),
    pve: new Signal(),
    eve: new Signal(),
    start: new Signal(),
    pass: new Signal(),
    finish: new Signal(),
    genpass: new Signal()
};

function add_board_size() {
    Signals.board_size.dispatch(2);
}

function sub_board_size() {
    Signals.board_size.dispatch(-2);
}

function change_player_color() {
    Signals.player_color.dispatch();
}

function add_komi() {
    Signals.komi.dispatch(0.5);
}

function sub_komi() {
    Signals.komi.dispatch(-0.5);
}

function pass() {
    Signals.pass.dispatch();
}

function pve() {
    Signals.pve.dispatch();
}

function eve() {
    Signals.eve.dispatch();
}

var EntryScene = (function () {
    var funcs = Halo.ResourceManager.funcs();

    funcs["add_board_size"] = add_board_size;
    funcs["sub_board_size"] = sub_board_size;
    funcs["change_player_color"] = change_player_color;
    funcs["add_komi"] = add_komi;
    funcs["sub_komi"] = sub_komi;
    funcs["pass"] = pass;
    funcs["pve"] = pve;
    funcs["eve"] = eve;
}());

// for editor
if (window.func_unloaders) {
    window.func_unloaders['main.js'] = function () {
        var funcs = Halo.ResourceManager.funcs();

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

function initBoard() {
    var scene = Halo.Config.scene();
    var root = scene.getWidgetRoot();
    var uiMain = Halo.Config.createWidget('main');
    var btnAddBoardSize = uiMain.getChild('btn_add_board_size');
    var txtBoardSize = uiMain.getChild('txt_board_size');
    var btnSubBoardSize = uiMain.getChild('btn_sub_board_size');
    var btnPlayerColor = uiMain.getChild('btn_player_color');
    var txtKomi = uiMain.getChild('txt_komi');
    var btnPass = Halo.Config.createWidget('pass');
    var txtInfo = Halo.Config.createWidget('info');
    var board = new Board();

    txtInfo.visible = false;
    root.add(uiMain);
    root.add(board.uiBoard);
    root.add(txtInfo);
    root.relayout();
    board.clear();

    Signals.board_size.add(function (s) {
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
    });

    Signals.player_color.add(function () {
        if (board.user === 'b') {
            board.user = 'w';
            btnPlayerColor.setSprite('go_map.png', 'white_png');
        } else if (board.user === 'w') {
            board.user = 'b';
            btnPlayerColor.setSprite('go_map.png', 'black_png');
        }
    });

    Signals.komi.add(function (k) {
        board.komi += k;
        txtKomi.text = 'Komi: ' + board.komi;
        txtKomi.refresh();
    });

    Signals.pve.add(function () {
        board.execute("boardsize " + board.size);
        board.execute("komi " + board.komi);
    });

    Signals.eve.add(function () {
        board.user = 'v';
        board.execute({
            cmd: 'start',
            board: {
                size: board.size,
                komi: board.komi
            }
        });
    });

    Signals.pass.add(function () {
        if (board.next === board.user) {
            board.execute({
                cmd: 'move',
                color: board.user,
                pass: true
            });
        }
    });

    Signals.start.add(function () {
        root.remove(uiMain);
        root.add(btnPass);
    });

    Signals.finish.add(function (score) {
        root.remove(btnPass);
        txtInfo.text = score;
        txtInfo.visible = true;
        txtInfo.refresh();
    });

    Signals.genpass.add(function () {
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
}());

// for editor
if (window.func_unloaders) {
    window.func_unloaders['main.js'] = function () {
        var funcs = Halo.ResourceManager.funcs();

        delete funcs["initBoard"];
    }
}