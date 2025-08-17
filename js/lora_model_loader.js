import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "comfyui-lora-model-loader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SilverLoraModelLoader") {
            nodeType.prototype.onExecuted = function(...args) {
                const nextModel = args[0]?.next_model?.[0];
                const currentRepeat = args[0]?.next_repeat?.[0];
                // Find the lora_model widget
                const loraModelWidget = this.widgets.find(w => w.name === "lora_name");
                const currentRepeatWidget = this.widgets.find(w => w.name === "current_repeat");
                currentRepeatWidget.value = currentRepeat;
                loraModelWidget.value = nextModel
            };
        }
    }
});
