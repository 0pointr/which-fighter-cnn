from fastai.learner import load_learner
import gradio as gr

models_names = ["fighters_single-cat_5+8-epochs.pkl",
"convnext-73pct-3+5+3.pkl",
"fighters-resnet50-cp-1-71pct.pkl",]

models = list(map(load_learner, [f"./models/{name}" for name in models_names]))

def get_categories(image_path):
    all_labels = {}
    for model in models:
        label,pred_idx,probs = model.predict(image_path)
        probs = probs.tolist()
        model_labeled_probs = dict([ [label,prob] for label,prob in zip(model.dls.vocab, probs) if prob > .1] )
        for label,prob in model_labeled_probs.items():
            all_labels.setdefault(label, []).append(prob)
    
    # print("before avg:", all_labels)
    for label,prob in all_labels.items():
        all_labels[label] = sum(prob) / len(prob)
    # print("after avg:", all_labels)

    return all_labels

def classify(image_path: str):
    labeled_probs = get_categories(image_path)
    print("\n".join( [f"{l}:{p:.3f}" for l,p in labeled_probs.items()]), "\n====\n" )
    return labeled_probs

demo = gr.Interface(fn=classify,
                    title="Which fighter aircraft is that?",
                    description="Upload/Paste photo of a fighter aircraft and I'll predict the model.",
                    inputs=gr.Image(width=512, height=512, type='filepath'),
                    outputs=gr.Label(num_top_classes=3),
                    flagging_mode='never',
                    allow_flagging='never')
demo.launch()
# print(label)
# print("\n".join(f"{cat} : {prob:.3f}" for cat,prob in zip(learner.dls.vocab, probs)) )
# learner.predict(a10)