var clock = new Halo.Clock();

function onResizeWindow() {
    var w = document.documentElement.clientWidth;
    var h = document.documentElement.clientHeight;

    Halo.Config.onResizeWindow(w, h);
}

function onMouseDown(evt) {
    var x = evt.clientX;
    var y = document.documentElement.clientHeight - evt.clientY;

    Halo.Config.scene() && Halo.Config.scene().handleEvent(evt, Halo.EVT_UI_DOWN, x, y);
}

function onMouseUp(evt) {
    var x = evt.clientX;
    var y = document.documentElement.clientHeight - evt.clientY;

    Halo.Config.scene() && Halo.Config.scene().handleEvent(evt, Halo.EVT_UI_UP, x, y);
}

function onMouseMove(evt) {
    var x = evt.clientX;
    var y = document.documentElement.clientHeight - evt.clientY;

    Halo.Config.scene() && Halo.Config.scene().handleEvent(evt, Halo.EVT_UI_Move, x, y);
}

var touch_identifier = null;

function onTouchStart(evt) {
    if (evt.changedTouches.length === 1) {
        var t = evt.changedTouches[0];
        var x = t.pageX;
        var y = document.documentElement.clientHeight - t.pageY;
        touch_identifier = t.identifier;
        Halo.Config.scene() && Halo.Config.scene().handleEvent(evt, Halo.EVT_UI_DOWN, x, y);
    } else {
        Halo.Config.scene() && Halo.Config.scene().handleEvent(evt);
    }
}

function onTouchEnd(evt) {
    if (evt.changedTouches.length === 1) {
        var t = evt.changedTouches[0];
        if (touch_identifier === t.identifier) {
            var x = t.pageX;
            var y = document.documentElement.clientHeight - t.pageY;
            Halo.Config.scene() && Halo.Config.scene().handleEvent(evt, Halo.EVT_UI_UP, x, y);
        }
    } else {
        Halo.Config.scene() && Halo.Config.scene().handleEvent(evt);
    }
}

function onTouchMove(evt) {
    if (evt.changedTouches.length === 1) {
        var t = evt.changedTouches[0];
        if (touch_identifier === t.identifier) {
            var x = t.pageX;
            var y = document.documentElement.clientHeight - t.pageY;
            Halo.Config.scene() && Halo.Config.scene().handleEvent(evt, Halo.EVT_UI_MOVE, x, y);
        }
    } else {
        Halo.Config.scene() && Halo.Config.scene().handleEvent(evt);
    }
}

function onKeyDown(evt) {
    Halo.Config.scene() && Halo.Config.scene().handleEvent(evt);
}

function onContextMenu(evt) {
    Halo.Config.scene() && Halo.Config.scene().handleEvent(evt);
}

function onMouseWheel(evt) {
    Halo.Config.scene() && Halo.Config.scene().handleEvent(evt);
}

function mainloop() {
    var dt = clock.getDelta();

    Halo.Config.update(dt);

    requestAnimationFrame(mainloop);
}

Halo.Config.bDebug = true;
var entryScene;

function init() {
    Halo.init('HaloGO', './assets/', null, function (res) {
        if (res === null) {
            window.addEventListener('resize', onResizeWindow, false);
            window.addEventListener('keydown', onKeyDown, false);
            document.addEventListener('contextmenu', onContextMenu, false);
            document.addEventListener('wheel', onMouseWheel, false);
            document.addEventListener('mousedown', onMouseDown, false);
            document.addEventListener('mouseup', onMouseUp, false);
            document.addEventListener('mousemove', onMouseMove, false);
            document.addEventListener('touchstart', onTouchStart, false);
            document.addEventListener('touchend', onTouchEnd, false);
            document.addEventListener('touchmove', onTouchMove, false);

            Halo.Config.start(document.body);
            Halo.Config.play(entryScene);

            onResizeWindow();
            mainloop();
        }
    });
}

function isWeixinBrowser(){
    var ua = navigator.userAgent.toLowerCase();
    return (/micromessenger/.test(ua)) ? true : false ;
}

if (isWeixinBrowser()) {
    document.addEventListener("WeixinJSBridgeReady", function () {
        init();
    }, false);

} else {
    document.onreadystatechange = function () {
        if (document.readyState === "complete") {
            init();
        }
    }
}