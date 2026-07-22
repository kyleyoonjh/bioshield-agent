# 치료 모달리티 플랫폼 확장 — 현황과 로드맵

> 이 문서는 여러 차례 대화에서 반복 제시된 "PLAN_THERAPEUTIC_MODALITY_PLATFORM" 스펙(v10.0 등)에
> 대한 응답으로 작성되었습니다. 해당 스펙들은 이 저장소에 실재한 적이 없었고, 그 안의 여러 항목
> (mRNA/항체 파이프라인, BindingDB·gnomAD·OpenTargets 연동, ML 예측 신뢰구간 등)은 실제로 구현되지도,
> 설계가 확정되지도 않았습니다. 이 문서는 그 내용을 "확정 사양"으로 재포장하는 대신, **실제로 무엇이
> 되어 있고 무엇이 되어 있지 않은지**를 정직하게 기록합니다. 아래 어떤 항목도 "완료"로 표시되지 않는 한
> 실제로 동작하지 않습니다.

---

## ✅ 실제로 구현·검증된 것 (Small Molecule 트랙 + 신항원 예측)

모두 `backend/services/`에 실제 코드가 있고, 이번 세션 중 라이브 API/실제 계산으로 검증되었습니다.

| 기능 | 실제 데이터 소스 | 파일 |
|---|---|---|
| 구조 확보 | AlphaFold DB / ESMFold (실시간) | `protein_structure_engine.py` |
| 리셉터 준비 + 블라인드 도킹 | AutoDock Vina (로컬 바이너리) | `receptor_prep_engine.py`, `docking_engine.py` |
| 스크리닝 병렬화 | Vina 프로세스 2개 동시 실행 (실측 ~1.8배 개선) | `drug_discovery_pipeline.py` |
| Plan→Execute→Reflect 재시도 | 최대 3회, 실행 품질 기준 | `drug_discovery_agent.py` |
| ADMET (일부) | Veber's Rule, PAINS(RDKit 공식 480종), SA score, 간독성 구조 경보 | `admet_engine.py` |
| 결합 포켓 분석 | 실제 도킹 포즈 좌표 + AlphaFold PAE | `structural_analysis_engine.py` |
| 문헌 검색 | PubMed 실시간 (esearch+efetch) | `literature_engine.py` |
| 임상시험 조회 | ClinicalTrials.gov 실시간 | `clinical_trials_engine.py` |
| 화합물 검색 | PubChem(유사 화합물), ChEMBL(실측 IC50) | `compound_discovery_engine.py` |
| 타겟 인텔리전스 | UniProt DISEASE 코멘트, Reactome 경로 | `target_intelligence_engine.py` |
| 구조 개선(SAR) | 실제 생물학적 등가체 치환 + 재도킹 비교 | `sar_optimization_engine.py`, `sar_optimization_service.py` |
| 종합 평가 | 투명한 공식의 우선순위 점수 + 근거 인용 | `decision_agent.py` |
| 체세포 변이 트랙 | VCF 업로드 + Ensembl VEP 실시간 재검증 | `vcf_annotation_engine.py` |
| 타겟-질병 연관성(OpenTargets) | OpenTargets Platform GraphQL API 실시간 (질병 연관 점수 + 소분자 druggability 등급) | `opentargets_engine.py` |
| 도킹 신뢰도(다중 포즈 RMSD 일관성) | Vina가 보고하는 대체 결합모드 간 RMSD 실측 — 블라인드 도킹 특유의 "여러 후보 부위" 모호성을 수치화 | `docking_engine.py::_compute_docking_confidence` |
| 타겟 우선순위 스코어링(도킹 전) | OpenTargets 연관성 + ChEMBL 기존 저해제 수를 조합한 투명 공식 — 풀 스크리닝 전에 타겟 유망도 사전 판단 | `target_intelligence_engine.py::calculate_target_priority_score`, `drug_discovery_chat.py` |
| 신항원 후보 예측(mRNA 암백신 설계의 앞단) | VCF→Ensembl VEP 실시간 재주석→실제 Ensembl 단백질 서열 조회→MHCflurry Class I 결합/제시 예측(공개 검증 에피토프로 사전 검증됨)→야생형 대비 이물성(foreignness) 비교. HLA는 BAM 기반 실제 타이핑이 아니라 인구집단 대표 6종 대립유전자 사용(OptiType은 Linux 전용 의존성 때문에 이 환경에 미설치 — 항상 그렇게 명시). 코돈 최적화/UTR 설계/mRNA 2차구조 폴딩은 별개로 여전히 없음(아래 ❌ 참고). | `neoantigen_engine.py`, `neoantigen_pipeline.py`, `vcf_annotation_engine.py` |

---

## 🔜 실제로 설계 가능한 다음 단계

이전 버전에서 이 섹션에 있던 세 항목(OpenTargets 실연동 / 도킹 신뢰도 / 타겟 우선순위 스코어링)은
모두 구현·검증되어 위 ✅ 표로 옮겼습니다. 현재 식별된 다음 단계는 없습니다 — 새로 하고 싶은 항목이
있으면 그때 이 섹션에 추가하고 범위를 논의합니다.

---

## ❌ 명시적으로 범위 밖 (그리고 그 이유)

| 항목 | 이유 |
|---|---|
| mRNA Therapeutics — 서열/구조 설계 (코돈 최적화, UTR 설계, mRNA 2차구조 폴딩, 단백질대체/사이토카인전달/유전자편집) | 완전히 별개의 과학 도메인, 관련 코드 전무. (단, 백신 설계의 앞단인 "신항원 후보 선별·면역원성 예측"은 위 ✅ 표에 실제 구현되어 있음 — 혼동 주의: mRNA 서열 자체를 설계하는 게 아니라 어떤 펩타이드를 표적으로 삼을지 고르는 단계.) |
| Antibody Discovery (에피토프/생성/구조예측/결합력) | CDR 설계, humanization, developability 등 별개 도메인. IgLM/AntiBERTa/DiscoTope/HADDOCK/DiffDock 중 이 환경에 설치·접근 가능한 것 없음. |
| BindingDB, gnomAD 연동 | 미연동. 계획 없음(요청 시 실제 연동 검토 가능). |
| Patent/FTO(특허 회피) 분석 | 법률 자문 영역 — 무료 API 부재, 잘못된 결과의 파급력이 커서 이 시스템 범위에서 제외. |
| GNINA/DiffDock 등 GPU 기반 도킹 | 이 환경에 GPU 없음. |
| hERG/CYP/BBB 등 임상 독성 예측 | 검증된 모델/무료 API 없음 — 지어내지 않음. |
| ML 기반 "예측 신뢰구간(prediction confidence)" | 이런 신뢰구간을 산출할 학습된 모델 자체가 없음. |

---

## 이 문서의 목적

이전 대화에서 반복 제시된 "확정 스펙"들은 실재하지 않는 파일 경로(`src/api/`, `src/engine/`)와
"모사 로그"로 명시된 가짜 도구 호출을 포함하고 있었습니다. 이 문서는 그 내용을 대체하는 것이며,
위 ❌ 표의 어떤 항목도 이 문서를 근거로 "이미 계획됨"으로 취급되어서는 안 됩니다 — 실제로 하고
싶은 항목이 있으면 그때 범위를 다시 논의합니다.
