
function EnterState_play() {
    if (this.instance === undefined) {
        if (this.params.target) {
            var nodes = this.params.target.split('.');
            var scope = this;

            switch (this.params.type) { //'O3': Object3D, 'W': Widget, 'M': Material, 'O2': Object2D
                case 'O3':
                    this.instance = Halo.Config.scene();
                    for (var i = 0; i !== nodes.length; ++i) {
                        this.instance = this.instance.getObjectByName(nodes[i]);
                        if (!this.instance) {
                            break;
                        }
                    }
                    break;
                case 'W':
                    this.instance = Halo.Config.scene().getWidgetRoot();
                    for (var i = 0; i !== nodes.length; ++i) {
                        this.instance.traverse(function (child) {
                            if (child.name === nodes[i]) {
                                scope.instance = child;
                                return false;
                            } else {
                                return true;
                            }
                        });
                    }
                default:
                    break;
            }
        }

        if (!this.instance) {
            this.instance = this.object;
        }
    }

    this.instance.enter(this.params.enter, this.params.flag);
}

function EnterState_stop() {
    if (this.reset) {
        this.instance.exit(this.params.enter, this.params.flag);
    }
}

(function () {
    var funcs = Halo.ResourceManager.funcs();
    funcs["EnterState_play"] = EnterState_play;
    funcs["EnterState_stop"] = EnterState_stop;
}());

//for editor
if (window.halo_func_params) {

    window.halo_func_params['EnterState.js'] = {
        type: 'O3',
        target: '',
        enter: '',
        flag: ''
    };

    window.halo_func_ui['EnterState.js'] = {
        type: function (block) {
            var logic = block.logic;

            var rType = new UI.Row().label('TargetType').select('type');
            rType.type.setOptions({ 'Ojbect3D': 'O3', 'Widget': 'W', 'Material': 'M', 'Object2D': 'O2' }).bind('change', function () {
                logic.params.type = rType.type.val();
                delete logic['instance'];
            });
            rType.type.val(logic.params.type);
            
            block.push(rType)
        },

        target: function (block) {
            var logic = block.logic;

            var rTarget = new UI.Row().label('Target').input('target', logic.params.target);
            rTarget.target.bind('change', function(){
                logic.params.target = rTarget.target.val();
                delete logic['instance'];
            });

            block.push(rTarget)
        }
    }
}