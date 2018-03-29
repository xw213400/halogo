
const Koa = require('koa');
const app = new Koa();
const session = require('koa-session');
const bodyParser = require('koa-bodyparser');
const router = require('koa-router')();
const fs = require('fs');
const request = require('request');
const path = require('path');
const serve = require('koa-static');
const Halogo = require('./Halogo');

const CONFIG = {
    key: 'koa:sess',
    maxAge: 'session',
    overwrite: true,
    httpOnly: true,
    signed: true,
    rolling: false,
};

app.keys = ['e6c74abd976cf4ddada0877b1a62615e'];
app.use(session(CONFIG, app));  // Include the session middleware

var halogo = new Halogo();

app.use(bodyParser({
    multipart: true,
    formLimit: 1000000000
}));

app.use(serve(__dirname + '/www'));

var root = router.get('/', function (ctx, next) {
    let n = ctx.session.views || 0;
    ctx.session.views = ++n;

    var html = fs.createReadStream(path.join(__dirname, "www/home.html"));
    ctx.type = 'html';
    ctx.body = html;
});

app.use(root.routes()).use(root.allowedMethods());

var execute = router.post('/execute', async function (ctx, next) {
    var msg = ctx.request.body;
    var halogo = null;
    if (msg.cmd === 'start') {
        halogo.clear();
    }

    await new Promise((resolve, reject) => {
        halogo.callback = function(res) {
            resolve(res);
        }
        halogo.execute(msg);
    }).then((res)=>{
        ctx.status = 200;
        ctx.response.message = res;
    }).catch((e)=>{
        console.error(e);
        ctx.status = 200;
        ctx.response.message = JSON.stringify({
            cmd: "unknow",
            ok: false,
            msg: 'Unknow error!'
        });
    });
});

app.use(execute.routes()).use(execute.allowedMethods());

// 在端口8008监听:
app.listen(8008);
console.log('app started at port 8008...');