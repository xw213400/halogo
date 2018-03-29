
const spawn = require('child_process').spawn;
const path = require('path');

var Halogo = function () {
    var scope = this;
    this.callback = null;

    var py = spawn('python', [path.join(__dirname, 'py/nodejs.py')]);
    var mid = 0;
    var cid = 0;
    var msgs = {};
    var cid_msg_map = {};

    py.stderr.on('data', function (data) {
        var str = data.toString();
        console.log('ERR:', str);
    });

    py.stdout.on('data', function (data) {
        var str = data.toString();
        console.log('DAT:', str);

        var args = str.split(/[\n\s]/);

        for (var i = 0; i !== args.length;) {
            if (args[i].length === 0) {
                args.splice(i, 1);
            } else {
                i++;
            }
        }

        var id = parseInt(args[0].substr(1));
        var msgdata = cid_msg_map[id];

        if (msgdata && msgdata.cmd_count > 0) {
            msgdata.cmd_count--;
            msgdata.ok = msgdata.ok && args[0].startsWith('=');
            if (msgdata.cmd_count === 0) {
                delete msgs[msgdata.mid];
                delete cid_msg_map[id];
                delete msgdata['mid'];
                delete msgdata['cmd_count'];
                if (msgdata.cmd === 'genmove' || msgdata.cmd === 'move') {
                    if (args[1] === 'pass') {
                        msgdata.pass = true;
                    } else {
                        var move = json_move(args[1]);
                        msgdata.i = move.i;
                        msgdata.j = move.j;

                        if (args.length >= 3) {
                            var caps = args[2].split(',');
                            msgdata.caps = [];
                            for (var i = 0; i != caps.length; ++i) {
                                msgdata.caps.push(json_move(caps[i]));
                            }
                        }
                    }
                } else if (msgdata.cmd === 'score') {
                    msgdata.score = args[1];
                }
                if (!msgdata.ok) {
                    msgdata.msg = args[1];
                }
                scope.callback(JSON.stringify(msgdata));
            }
        }

        console.log('RET:', msgdata);
    });

    function json_move(gtp) {
        var i = gtp.charCodeAt(0) - 65;
        if (i >= 9) {
            i--;
        }
        j = parseInt(gtp.substr(1)) - 1;

        return { i: i, j: j };
    }

    function gtp_move(i, j) {
        if (i >= 8) {
            i++;
        }
        return String.fromCharCode(65 + i, 49 + j);
    }

    function post_cmd(msgdata) {
        cid++;
        msgdata.cmd_count++;
        cid_msg_map[cid] = msgdata;
        var args = [cid.toString()];
        args.push.apply(args, Array.prototype.slice.call(arguments, 1));
        var cmd = args.join(' ');
        console.log('CMD:', cmd);
        py.stdin.write(cmd + '\n');
    }

    this.execute = function (msg) {
        mid++;
        var msgdata = msgs[mid];
        if (msgdata === undefined) {
            msgdata = msgs[mid] = { cmd: msg.cmd, ok: true, mid: mid, cmd_count: 0 };
        }
        switch (msg.cmd) {
            case 'start':
                post_cmd(msgdata, 'boardsize', msg.board.size);
                post_cmd(msgdata, 'komi', msg.board.komi);
                break;
            case 'move':
                if (msg.pass) {
                    post_cmd(msgdata, 'play', msg.color, 'pass');
                } else {
                    post_cmd(msgdata, 'play', msg.color, gtp_move(msg.i, msg.j));
                }
                break;
            case 'genmove':
                post_cmd(msgdata, 'genmove', msg.color);
                break;
            case 'score':
                post_cmd(msgdata, 'score');
                break;
            default:
                break;
        }
    };
};

module.exports = Halogo;