# a case of drs-text generation
from tokenization_mlm import MLMTokenizer
from transformers import MBartForConditionalGeneration


if __name__ == '__main__':
    # For DRS parsing, src_lang should be set to en_XX, de_DE, it_IT, or nl_XX
    tokenizer = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='<drs>')
    model = MBartForConditionalGeneration.from_pretrained('laihuiyuan/DRS-LMM')

    # gold text: The court is adjourned until 3:00 p.m. on March 1st.
    inp_ids = tokenizer.encode(
        "court.n.01 time.n.08 EQU now adjourn.v.01 Theme -2 Time -1 Finish +1 time.n.08 ClockTime 15:00 MonthOfYear 3 DayOfMonth 1",
        return_tensors="pt")

    # For DRS parsing, the forced bos token here should be <drs>
    foced_ids = tokenizer.encode("en_XX", add_special_tokens=False, return_tensors="pt")
    outs = model.generate(input_ids=inp_ids, forced_bos_token_id=foced_ids.item(), num_beams=5, max_length=150)
    text = tokenizer.decode(outs[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(1)