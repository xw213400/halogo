
const spawn = require('child_process').spawn;
const path = require('path');

var Halogo = function () {
    var py = spawn('python', [path.join(__dirname, 'py/nodejs.py')]);
    var onsuccess = null;
    var onerror = null;

    py.stderr.on('data', function (data) {
        onerror && onerror(data.toString());
    });

    py.stdout.on('data', function (data) {
        var msg = data.toString().split(/[\n\s]/);
        for (var i = 0; i !== msg.length;) {
            if (msg[i] === '') {
                msg.splice(i, 1);
            } else {
                ++i;
            }
        }
        onsuccess && onsuccess(msg);
    });

    this.execute = function (cmd) {
        console.log("X:", cmd);

        var promise = new Promise((resolve, reject) => {
            onsuccess = function (msg) {
                resolve(msg);
            };
            onerror = function(err) {
                reject(err);
            }
        });

        py.stdin.write(cmd + '\n');

        return promise;
    };
};

module.exports = Halogo;