var board;
var records;
var step;
var wt_board_info;

function update_step() {
    wt_board_info.text = (step+1).toString() + ' in ' + records.length.toString();
    wt_board_info.refresh();
    board.setBoard(records[step].bb);
}

function do_last_step(ds) {
    step -= ds;
    if (step < 0) {
        step = 0;
    }
    update_step();
}

function do_next_step(ds) {
    step += ds;
    if (step >= records.length-1) {
        step = records.length-1;
    }
    update_step();
}

function debug_play() {
    var flag = this.parent.flag;

    if (flag === 'last') {
        do_last_step(1);
    } else if (flag === 'next') {
        do_next_step(1);
    } else if (flag === 'last10') {
        do_last_step(10);
    } else if (flag === 'next10') {
        do_next_step(10);
    } else {
        wt_board_info = Halo.Config.scene().getWidgetRoot().getChild('debug').getChild('board_info');

        board = new Board();
    
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
}


var debug = (function () {
    var funcs = Halo.ResourceManager.funcs();

    funcs["debug_play"] = debug_play;
}());