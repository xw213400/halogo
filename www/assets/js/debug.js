var board;
var records;
var step;
var wt_board_info;

function update_step() {
    wt_board_info.text = (step+1).toString() + ' in ' + records.length.toString();
    wt_board_info.refresh();
    board.setBoard(records[step].bb);
}

function on_last_step() {
    step--;
    if (step < 0) {
        step = 0;
    }
    update_step();
}

function on_next_step() {
    step++;
    if (step >= records.length-1) {
        step = records.length-1;
    }
    update_step();
}

function debug_init(scene) {
    var ui = scene.getWidgetRoot();
    
    wt_board_info = ui.getChild('debug').getChild('board_info');

    board = new Board();
    ui.add(board.uiBoard);
    board.clear();

    Halo.httpRequest('res/record.json', 'text').then((data) => {
        records = JSON.parse(data);
        for (var i = 0; i !== records.length; ++i) {
            var record = records[i];
            record.bb = [];
            for (var j = 0; j !== record.board.length; ++j) {
                var c = record.board[j];
                if (c !== 2) {
                    record.bb.push(c);
                }
            }
        }

        step = Math.floor(records.length / 2);
        var record = records[step];
        board.size = Math.sqrt(record.bb.length);
        board.clear();
        update_step();
    });
}


var debug = (function () {
    var funcs = Halo.ResourceManager.funcs();

    funcs["debug_init"] = debug_init;
    funcs["on_last_step"] = on_last_step;
    funcs["on_next_step"] = on_next_step;
}());

if (window.func_unloaders) {
    window.func_unloaders['debug.js'] = function () {
        var funcs = Halo.ResourceManager.funcs();

        delete funcs["debug_init"];
        delete funcs["on_last_step"];
        delete funcs["on_next_step"];
    }
}