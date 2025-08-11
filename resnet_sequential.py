"""
This code was written with reference to SparseGPT and GPTQ.  
Modified on: 2025-07-30  
Created by: Seokho Han
"""
import torch
import torch.nn as nn
from Quantizer import Quantizer
from CombinedCompressor import CombinedCompressor, start_global_progress, close_global_progress
from utils import find_layers_resnet

@torch.no_grad()

def resnet_sequential(model, calib_loader, device, layer_configs, params):
    layers = find_layers_resnet(model)
    # 글로벌 진행바를 레이어 개수 기준으로 초기화 (Kaggle에서 셀 재실행 시에도 매번 초기화)
    gbar = start_global_progress(total_layers=len(layers), desc="Pruning All Layers")

    for idx, (name, module) in enumerate(layers):
        # 설정이 없으면 default 값 사용
        config = layer_configs.get(name, {})
        sparsity = config.get('sparsity', params["DEFAULT_SPARSITY"])
        wbits = config.get('wbits', params["DEFAULT_WBITS"])

        # 레이어명은 tqdm desc에 띄우지 않고, 로그로만 출력
        gbar.note(f"[{idx+1}/{len(layers)}] Processing {name} | Sparsity: {sparsity}, W_Bits: {wbits}")

        # 압축을 수행할 필요가 없는 경우 (희소성 0, 16비트 양자화)
        if sparsity == 0 and wbits >= 16:
            gbar.note(f"  -> Skipping compression for {name}.")
            gbar.update(1)
            continue

        module.to(device)

        gpt = CombinedCompressor(module)

        if wbits < 16:
            gpt.quantizer = Quantizer()
            gpt.quantizer.configure(bits=wbits, perchannel=True, sym=False, mse=False)

        cache = {}
        def save_io(mod, inp, out):
            cache['inp'] = inp[0].detach()
            cache['out'] = out.detach()

        handle = module.register_forward_hook(save_io)

        for batch_idx, (img, _) in enumerate(calib_loader):
            model.to(device)
            model(img.to(device))
            if 'inp' in cache and 'out' in cache:
                gpt.add_batch(cache['inp'], cache['out'])
            if batch_idx >= params['nsamples'] - 1:
                break

        handle.remove()

        # 프루닝 및 양자화 실행
        gpt.fasterprune(sparsity=sparsity, prunen=params['prunen'], prunem=params['prunem'],
                       percdamp=params['percdamp'], blocksize=params['blocksize'])
        gpt.free()

        cache.clear()
        module.cpu()
        torch.cuda.empty_cache()
    # 전체 완료 후 글로벌 바 닫기 (다음 셀 재실행 시 ASCII 및 바 재출력 보장)
    close_global_progress()
