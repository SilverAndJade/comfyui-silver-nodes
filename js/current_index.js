import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "comfyui-current",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (["SilverFileTextLoader", "SilverFolderImageLoader", "SilverFolderVideoLoader", "SilverFolderFilePathLoader"].indexOf(nodeData.name) > -1) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(...args) {
                onExecuted?.apply(this, [message]);
                debugger;
                const currentIndex = args[0].next_index[0];
                const currentIndexWidget = this.widgets.find(w => w.name === "current_index")
                if (typeof currentIndex === "number" && currentIndexWidget) {
                    currentIndexWidget.value = currentIndex;
                }
            };
        }
    }
});
