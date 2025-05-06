from fastai.learner import load_learner
import gradio as gr

model_path = "fighters_single-cat_5+8-epochs.pkl"
learner = load_learner(model_path)

def classify(image_path: str):
    label,pred_idx,probs = learner.predict(image_path)
    labeled_probs = dict([ [label,prob] for label,prob in zip(learner.dls.vocab, probs) if prob > .1] )
    print("\n".join( [f"{l}:{p:.3f}" for l,p in labeled_probs.items()]) )
    return labeled_probs

demo = gr.Interface(fn=classify,
                    title="Which fighter is that?",
                    inputs=gr.Image(width=512, height=512, type='filepath'),
                    outputs=gr.Label(num_top_classes=3),
                    flagging_mode='never',
                    allow_flagging='never')
demo.launch()
# print(label)
# print("\n".join(f"{cat} : {prob:.3f}" for cat,prob in zip(learner.dls.vocab, probs)) )
# learner.predict(a10)