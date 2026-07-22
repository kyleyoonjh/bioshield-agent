# 심층 신항원 분석 및 임상적 타당성 리포트 (샘플 데이터 기반)

**대상 변이**: KRAS G12D (chr12:25245350 C>T, GRCh38) · **입력 데이터**: `sample/NSCLC_variants.vcf` + `sample/NSCLC.bam` (합성 샘플 데이터) · **생성 시각 기준 파이프라인**: AiRemedy Drug Discovery Assistant — 신항원 후보 식별 모듈

> ⚠️ **최상위 고지사항 (반드시 먼저 읽어주세요)**
> 이 리포트는 **실제 환자 데이터가 아닌 합성 샘플 데이터**를 기반으로 생성되었으며, 아래 3가지 사항이 원 요청과 다르게 작성되었습니다 (지어내지 않기 위한 의도적 수정):
> 1. **Foreignness 계산법을 정정했습니다** — 원 요청은 "BLOSUM62 매트릭스 기반 공식"을 설명해 달라고 하셨지만, 본 파이프라인은 BLOSUM62를 전혀 사용하지 않았습니다. 실제로 사용한 방법(야생형 대비 MHCflurry percentile 차이)만 정확히 기술합니다.
> 2. **"임상 성공 가능성" 점수 요소를 제외했습니다** — 이 변이/환자에 대한 실제 임상 성공률 데이터가 존재하지 않습니다. 점수에 포함하면 숫자를 지어내는 것이므로, 실제 계산 가능한 3개 요소(결합친화도·제시확률·foreignness)만으로 점수를 구성했습니다.
> 3. **"추천/보류/기각" 최종 Verdict를 제공하지 않습니다** — 샘플 데이터와 가상 population-common HLA allele만으로는 실제 임상 채택 여부를 판단할 근거가 없습니다. 대신 실제로 필요한 추가 검증 항목을 명시합니다.

---

## 1. 기술 및 알고리즘 아키텍처 상세 설명

### 1.1 체세포 변이 필터링 (WES/RNA-Seq Tumor-Normal Paired Analysis) — 일반 방법론 설명

실제 임상 파이프라인에서 종양-정상 쌍체(Tumor-Normal Paired) WES/RNA-Seq 데이터로부터 체세포 변이(Somatic Mutation)를 호출할 때 표준적으로 쓰이는 개념은 각 유전체 좌표에서 종양 샘플의 관측 데이터가 "체세포 변이" 가설과 "생식세포 변이 또는 시퀀싱 오류" 가설 중 어느 쪽을 더 강하게 지지하는지를 우도비(likelihood ratio)로 판정하는 것입니다 (MuTect2 등에서 쓰이는 개념):

```
LOD_somatic = log10( P(reads | 체세포 변이 모델) / P(reads | 생식세포 변이·시퀀싱 오류 모델) )
```

여기서 각 모델은 해당 위치의 대립유전자 빈도(Variant Allele Frequency, VAF), 염기 정확도(base quality), 스트랜드 편향(strand bias) 등을 반영하며, LOD 값이 사전 정의된 임계치(예: MuTect2 기본값 6.3)를 넘을 때 체세포 변이로 확정합니다. 정상 조직 샘플과의 대조를 통해 생식세포 다형성(SNP)을 제외하는 것이 핵심입니다.

**⚠️ 실제 실행 여부**: 본 파이프라인은 이 계산을 직접 수행하지 않았습니다. 입력된 `sample/NSCLC_variants.vcf`는 이미 변이 호출이 완료된 VCF 파일이며, 본 시스템이 실제로 수행한 것은 그 위에 **Ensembl VEP를 통한 실시간 재주석**(`services/vcf_annotation_engine.py`)뿐입니다 — VCF 파일 자체의 `INFO=GENE` 라벨을 신뢰하지 않고, 정확한 유전체 좌표에서 실제 단백질 코딩 결과(아미노산 변화·transcript ID)를 Ensembl 서버에 실시간으로 재확인하는 방식입니다. 원시 BAM/FASTQ로부터의 정렬·변이 호출 자체는 이 시스템의 범위 밖입니다.

### 1.2 MHCflurry의 결합친화도·제시확률 예측 원리

MHCflurry(Class1PresentationPredictor, 이 파이프라인이 실제로 로컬에서 실행한 사전학습 모델)는 두 개의 하위 예측기를 결합한 앙상블 신경망 구조입니다:

- **결합친화도 예측기(Affinity Predictor)**: 각 HLA allele별로 독립적으로 학습된 순전파 신경망(feed-forward neural network)이 펩타이드 서열을 입력받아 IC50 결합친화도(nM)를 회귀 예측합니다. 학습 데이터는 IEDB(Immune Epitope Database)의 실측 결합 어세이 데이터이며, 출력값은 `1 - log50000(IC50)` 변환(NetMHC 계열에서 표준적으로 쓰이는 정규화 방식)을 통해 0~1 스케일로 변환된 뒤 다시 nM 단위로 역변환되어 보고됩니다.
- **항원 프로세싱 예측기(Processing Predictor)**: 프로테아좀 절단(proteasomal cleavage)과 TAP(transporter associated with antigen processing) 수송 효율을 근사하는 별도 신경망으로, 실제 세포 내에서 해당 펩타이드가 MHC-I 경로까지 도달할 가능성을 반영합니다.
- **최종 제시확률(Presentation Score)**: 위 두 예측기의 출력을 결합해 산출되며, 실제 질량분석 기반 HLA 리간돔(immunopeptidomics mass-spec) 데이터로 보정되어 있습니다. `presentation_percentile`은 이 제시확률을 동일 allele에 대한 무작위 펩타이드 배경분포와 비교한 백분위 순위입니다(수치가 낮을수록 강한 결합·높은 제시 가능성 — 표준 NetMHCpan/MHCflurry 관례).

본 파이프라인이 실제로 호출한 함수는 `Class1PresentationPredictor.predict(peptides, alleles)`이며, 사전 검증 시 실제 잘 알려진 에피토프(NLVPMVATV/CMV pp65 → 16.6 nM, GILGFVFTL/flu M1 → 20 nM, 둘 다 강한 결합 — 실제 면역학적 지식과 일치)로 정확성을 확인했습니다.

### 1.3 "비자기 인식 잠재력(Foreignness)" — 실제 계산 방법 (정정)

**실제로 사용한 공식**:

```
Foreignness = Percentile(야생형 펩타이드) − Percentile(변이 펩타이드)
```

동일한 위치·길이의 펩타이드 윈도우를 변이형(mutant)과 야생형(wildtype) 두 버전으로 만들어, **동일한 HLA allele 조합에 대해 MHCflurry로 각각 독립적으로 예측**한 뒤 percentile 차이를 구합니다. 이 값이 클수록 "변이로 인해 새롭게 강하게 제시되지만, 원래(야생형) 서열은 면역계가 거의 인식하지 못했을 펩타이드"라는 뜻이며, 이는 T세포가 이미 관용(tolerance)되어 있지 않을 가능성이 높다는 것을 시사하는 실제적이고 해석 가능한 지표입니다.

**BLOSUM62에 대한 정정**: 원 요청에서 언급하신 BLOSUM62(BLOcks SUbstitution Matrix)는 아미노산 치환의 진화적 유사도를 점수화하는 실제 존재하는 행렬이며, 일부 발표된 신항원 파이프라인(예: Balachandran et al. 2017의 "Neoantigen Fitness Model", TCR 교차반응성 예측에 물리화학적 유사도를 쓰는 방법론)에서 self-peptide와의 서열 유사도 페널티를 계산하는 데 실제로 사용됩니다. 그러나 **본 분석에서는 BLOSUM62를 전혀 사용하지 않았습니다** — 사용했다고 서술하면 실행되지 않은 방법론을 실행된 것처럼 기술하는 것이 되므로, 정정하여 명시합니다. 전체 인체 정상 단백질체(self-proteome) 대상 BLAST 비교 역시 본 분석 범위에 포함되지 않았습니다(위치 특이적 야생형 대조군 비교로 범위를 한정).

---

## 2. 환자 데이터 종합 요약 테이블

### 분석 환경 제약 사항 (필독)

| 제약 사항 | 내용 |
|---|---|
| **HLA 타이핑 미수행** | 실제 BAM 기반 HLA 타이핑(OptiType 등)은 Docker/WSL 등 Linux 환경이 필요해 이 배포 환경에서 실행되지 않았습니다. 아래 결과는 **이 환자의 실제 유전형이 아니라**, 인구집단 고빈도 HLA class I allele 6종(HLA-A\*02:01, HLA-A\*01:01, HLA-B\*07:02, HLA-B\*08:01, HLA-C\*07:01, HLA-C\*07:02)을 대신 사용한 결과입니다. |
| **BAM의 실제 용도** | 업로드된 BAM은 실제로 파싱되어 헤더/리드 수(42개, chr12 참조)를 확인하는 데만 사용되었으며, HLA 타이핑에는 사용되지 않았습니다. |
| **입력 데이터 성격** | `sample/NSCLC_variants.vcf`·`sample/NSCLC.bam`은 실제 hg38 참조서열 기반으로 구성된 **합성 샘플 데이터**이며, 실제 환자 유래 데이터가 아닙니다. |
| **자기 단백질체 비교 범위** | 전체 self-proteome BLAST가 아닌, 해당 위치의 야생형 대조 서열과의 비교로 한정됩니다 (1.3절 참고). |

### 변이 및 신항원 후보 요약

| 항목 | 값 |
|---|---|
| 유전자 | KRAS |
| 아미노산 변화 | G12D (c.35G>A, transcript ENST00000256078) |
| 유전체 좌표 (GRCh38) | chr12:25245350 C>T |
| Variant Allele Frequency (VCF 기재값) | 0.37 |
| 변이 펩타이드 (8-mer) | `DGVGKSAL` |
| 야생형 대응 펩타이드 | `GGVGKSAL` |
| 결합 친화도 (변이) | 489.1 nM |
| 결합 친화도 (야생형) | 7,819.9 nM |
| 제시 백분위수 (변이) | 0.938 (percentile — 강한 결합 기준 ≤2.0 충족) |
| 제시 백분위수 (야생형) | 16.158 (percentile — 비자기 기준 >10.0 충족) |
| Foreignness (야생형 percentile − 변이 percentile) | 15.219 |
| 최적 결합 Allele | HLA-B\*08:01 (population-common, 실제 환자 유전형 아님) |
| 강한 결합 여부 | 예 (percentile ≤ 2.0) |
| 자기유사성(Self-similarity) 여부 | 아니오 (야생형 percentile > 10.0) |

---

## 3. AI 에이전트 자체 종합 점수 매트릭스 (AI Neo-Score)

### 산출 로직 (실제 코드로 구현·실행됨: `services/neoantigen_engine.py::calculate_neoantigen_composite_score`)

100점 만점을 아래 **3개의 실제 계산 가능한 요소**로만 구성합니다. "임상 성공 가능성"은 포함하지 않습니다 — 이 변이·환자에 대한 실제 임상 성공률 데이터가 존재하지 않아, 포함할 경우 수치를 지어내는 것이 되기 때문입니다.

| 요소 | 가중치 | 계산 방법 | 근거 |
|---|---|---|---|
| 결합친화도 (Affinity) | 30점 | 50 nM(만점 기준)~500 nM(0점 기준) 구간을 로그 스케일로 정규화 | 실제 MHCflurry IC50 예측값 |
| 제시확률 (Presentation) | 40점 | percentile 0(만점)~2.0(0점, 강한 결합 임계치) 구간 선형 정규화 | 실제 MHCflurry presentation_percentile |
| 비자기 신선도 (Foreignness) | 30점 | foreignness 값 자체를 0~30 구간에 캡핑 | 위 1.3절의 실제 percentile 차이 |

```
AI Neo-Score = affinity_component + presentation_component + foreignness_component
```

### 이 후보의 실제 산출 결과 (코드 실행, 지어낸 값 아님)

| 구성 요소 | 점수 | 산출 근거 |
|---|---|---|
| 결합친화도 | **0.3 / 30.0** | 489.1 nM — "강한 결합" 표준 임계치인 500 nM에 매우 근접해, 로그 스케일 정규화상 낮은 점수 |
| 제시확률 | **21.2 / 40.0** | percentile 0.938 — 강한 결합 기준(≤2.0)은 충족하나 최상위권(0.5 이하)에는 못 미침 |
| Foreignness | **15.2 / 30.0** | 실제 percentile 차이 15.219, 중간 수준 |
| **합계 (AI Neo-Score)** | **36.7 / 100.0** | |

### 해석 (판정이 아닌 서술)

36.7/100은 3개 하위 지표 중 결합친화도가 특히 "강한 결합"과 "약한 결합"의 경계선(489.1 nM vs 500 nM 임계치)에 위치해 전체 점수를 끌어내리는 구조입니다. 제시확률과 foreignness는 상대적으로 양호합니다. **이 숫자만으로 "채택/보류/기각"을 판단하지 않습니다** — 아래는 이 숫자가 아니라, 이 파이프라인의 실제 한계에서 비롯된 정직한 다음 단계입니다:

- 실제 환자 유래 BAM에 대한 진짜 HLA 타이핑(OptiType 등, Docker/WSL 환경 필요) 없이는 HLA-B\*08:01이 이 환자의 실제 대립유전자인지조차 알 수 없습니다.
- 실제 T세포 반응성 검증(ELISPOT, tetramer assay 등 습식 실험)이 이 파이프라인 범위 밖입니다 — MHCflurry 예측은 "제시 가능성"이지 "면역원성"의 직접 증명이 아닙니다.
- 전체 self-proteome 대상 교차반응성(cross-reactivity) 스크리닝이 수행되지 않았습니다.

---

## 4. PubMed 문헌 기반 심층 교차 검증 (RAG Deep Dive)

*아래는 이전 대화에서 실제로 PubMed API를 통해 조회한 3편의 실제 논문(PMID 36027916, 33016924, 38195752)의 실제 초록 내용에 근거합니다. 초록에 없는 세부 수치·결론은 추가하지 않습니다.*

### 4.1 면역학적 기회 (Opportunity)

**Pant et al. 2024, Nature Medicine (PMID: 38195752, AMPLIFY-201 1상)**: 림프절 표적 amphiphile 변형 KRAS G12D/G12R 펩타이드 백신(ELI-002 2P)이 mKRAS 미세잔존질환 양성 환자 25명(췌장암 20명, 대장암 5명) 중 **21명(84%)에서 실제 ex vivo mKRAS 특이 T세포 반응**을 유도했고, 21명(84%)에서 종양 바이오마커 반응이 관찰되었습니다. T세포 반응 강도(기저 대비 12.75배 초과 증가군)와 임상 지표(종양 바이오마커 감소율 −76.0% vs −10.2%, 무재발생존 기간 중앙값 미도달 vs 4.01개월)가 유의하게 상관했습니다(HR=0.14, P=0.0167). 이는 KRAS G12D 표적 백신이 **실제로 측정 가능한 임상적 이득과 연결될 수 있다는 실제 근거**입니다.

**Awad et al. 2022, Cancer Cell (PMID: 36027916, NEO-PV-01)**: 개인 맞춤형 신항원 백신이 non-squamous NSCLC 1차 치료에서 화학요법·항PD-1과 병용되었을 때, de novo 신항원 특이 CD4+/CD8+ T세포 반응을 유도했으며, **백신에 포함되지 않은 KRAS G12C/G12V 변이에 대한 반응으로까지 에피토프가 확산(epitope spread)**되는 현상이 관찰되었습니다. 이는 KRAS 신항원 표적 백신이 단일 에피토프를 넘어 광범위한 항종양 면역을 촉발할 잠재력을 시사합니다.

### 4.2 면역학적 위험 요소 및 병용요법의 필요성 (Risk & Combination Rationale)

**Cafri et al. 2020, Journal of Clinical Investigation (PMID: 33016924)**: 전이성 위장관암 환자 4명에게 검증된 신항원(KRAS G12D 포함)을 인코딩한 mRNA 백신을 투여한 결과, **백신 접종 전에는 검출되지 않았던 변이 특이 T세포 반응이 유도**되었고 KRAS G12D를 표적하는 T세포 수용체(TCR)를 실제로 분리·검증하는 데 성공했습니다. 그러나 **4명의 치료 환자 중 객관적 임상 반응은 관찰되지 않았습니다.**

이 결과는 "면역원성(T세포 유도)"과 "임상적 유효성(종양 축소)" 사이에 실질적인 간극이 존재함을 보여주는 실제 데이터입니다. 개발자/연구원 관점에서 이 간극을 설명할 수 있는 면역미세환경(Tumor Microenvironment, TME) 기전은 다음과 같습니다:

1. **T세포 소진(T-cell exhaustion)**: 종양 미세환경 내에서 신항원 특이 T세포가 만성적 항원 노출과 억제성 신호(PD-1/PD-L1, CTLA-4, LAG-3 등 immune checkpoint 축)에 지속적으로 노출되면, 살해능(cytotoxicity)이 저하된 소진 표현형(exhausted phenotype)으로 전환됩니다. 백신으로 말초에서 T세포를 유도하더라도, 종양 부위에 침윤한 뒤 이 억제 신호에 곧바로 무력화될 수 있습니다.
2. **면역억제성 종양 미세환경**: 골수유래억제세포(MDSC), 조절 T세포(Treg), M2형 종양연관대식세포(TAM) 등이 종양 국소에서 T세포 기능을 억제하는 물리적·생화학적 장벽을 형성합니다 — Pant et al.(PMID 29860986, 앞선 대화에서 조회)이 지적한 췌장암의 "면역 억제 환경으로 인한 면역치료 효과 제한"과 같은 맥락입니다.
3. **T세포 침윤 자체의 한계**: 백신이 순환계(peripheral blood)에서 T세포를 유도해도, 종양 실질로의 실제 침윤(infiltration)이 이루어지지 않으면 임상 반응으로 이어지지 않습니다.

**이 세 가지 기전이 바로 면역관문억제제(예: 키트루다/Pembrolizumab, PD-1 억제제) 병용요법이 실질적으로 요구되는 이유입니다**: 백신이 신항원 특이 T세포의 "양(quantity)"을 늘리는 역할을 한다면, 면역관문억제제는 이미 종양에 도달한 T세포가 소진되지 않고 실제 살해 기능을 발휘하도록 "질(quality)"을 보존하는 역할을 합니다. 실제로 Awad et al.(PMID 36027916)의 NEO-PV-01 시험 자체가 이미 백신을 항PD-1(pembrolizumab)과 화학요법에 처음부터 병용하는 설계였다는 점, 그리고 Rappaport et al.(PMID 38538867, 앞선 대화에서 조회)의 KRAS 신항원 백신이 ipilimumab(항CTLA-4)+nivolumab(항PD-1) 면역관문억제제 병용으로 설계되었다는 점은, 실제 임상 개발 현장에서 이미 "백신 단독으로는 불충분하다"는 판단이 반영되어 있음을 보여주는 실제 근거입니다.

**결론적으로**, 이 리포트의 KRAS G12D/DGVGKSAL 후보를 실제 mRNA 자가 백신 개발로 이어가려면, Cafri et al.의 사례가 보여주듯 **백신 단독요법보다 면역관문억제제 병용을 처음부터 시험 설계에 포함하는 것이 문헌적으로 뒷받침되는 접근**이며, 이는 이 파이프라인의 계산 결과가 아니라 인용된 3편(및 앞선 대화에서 조회된 2편)의 실제 논문 내용에 근거한 서술입니다.

---

## 부록: 이 리포트에서 의도적으로 제공하지 않은 항목

- **최종 채택/보류/기각 Verdict**: 샘플 데이터 + 가상 HLA로는 실제 임상 개발 의사결정을 내릴 근거가 없습니다.
- **임상 성공 확률 수치**: 존재하지 않는 데이터를 만들어내는 것이므로 제공하지 않습니다.
- **BLOSUM62 기반 foreignness 값**: 실제로 계산하지 않았으므로 제공하지 않습니다.
