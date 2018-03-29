
const Koa = require('koa');
const app = new Koa();
const bodyParser = require('koa-bodyparser');
const router = require('koa-router')();
const fs = require('fs');
const request = require('request');
const path = require('path');
const serve = require('koa-static');
const Halogo = require('./halogo');


var halogo = new Halogo();

app.use(bodyParser({
    multipart: true,
    formLimit: 1000
}));

app.use(serve(__dirname + '/www'));

var root = router.get('/', function (ctx, next) {
    var html = fs.createReadStream(path.join(__dirname, "www/home.html"));
    ctx.type = 'html';
    ctx.body = html;
});

app.use(root.routes()).use(root.allowedMethods());

var execute = router.post('/execute', async function (ctx, next) {
    var cmd = ctx.request.body;

    await halogo.execute(cmd.join(' ')).then((msg) => {
        console.log("R:", msg);
        ctx.status = 200;
        ctx.response.message = JSON.stringify(msg);
    }).catch((err) => {
        console.error("E:", err);
        ctx.status = 200;
        ctx.response.message = err;
    });
});

app.use(execute.routes()).use(execute.allowedMethods());

// 在端口8008监听:
app.listen(8008);
console.log('app started at port 8008...');