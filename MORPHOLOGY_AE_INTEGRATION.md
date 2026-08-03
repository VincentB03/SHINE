# Intégration du modèle génératif (AutoEncoder + Flow) — plan d'action

Objectif : remplacer le rendu paramétrique (`galsim.Exponential` + priors flux/hlr/e1/e2)
**uniquement pour le tier de stamps 64×64 px**, en réutilisant l'AutoEncoder + Normalizing
Flow entraînés dans `Train-AE` (package `pshear`, sur le Bureau). Les tiers 128 et 256 px
continuent d'utiliser le renderer Sersic/Exponential existant, inchangé.

Source du modèle entraîné : `~/Desktop/Train-AE`
- `pshear/galaxy.py` → `GalaxyAutoEncoder` (encode → latent `z` → decode → image `g` →
  convolution PSF)
- `pshear/nn/flow.py` → `LatentFlow` (prior normalizing flow sur `z`, actuellement
  inconditionnel)
- `pshear/utils.py` → `load_galaxy_autoencoder()`, `load_flow()` (chargement des
  checkpoints `.eqx` + `config.yaml`)

Pipeline cible : `shine/euclid/scene.py` (`MultiExposureScene`), tiers gérés par
`galaxy_stamp_sizes` dans `shine/euclid/config.py`.

---

## Principe du modèle hybride

```
tier 64 px  (petites/faibles galaxies) → z ~ flow.prior ; g = AE.decode(z) ; shear(g) ; conv(PSF réel)
tier 128 px (galaxies moyennes)        → inchangé : galsim.Exponential(flux, hlr).shear(e).shear(g)
tier 256 px (grandes galaxies)         → inchangé : idem
```

Seul le **shear global (g1, g2)** reste un paramètre physique à inférer pour les sources du
tier appris ; `flux`, `hlr`, `e1`, `e2` disparaissent pour ce tier (implicitement encodés
dans `z`).

---

## Étape 0 — Pré-requis

- [ ] **Dépendances** : ajouter à `pyproject.toml` (`dependencies`) :
  ```toml
  "equinox>=0.11",
  "flowjax>=..." ,
  "paramax",
  "einops",
  ```
  Vérifier les versions exactes utilisées dans `Train-AE/requirements.txt` et les aligner.
- [ ] **Checkpoints** : identifier un run d'AE (`ae_run_dir`, `ae_epoch`) et de flow
  entraînés **sur des stamps 64×64 à l'échelle 0.1"/px** (config `nx=64, ny=64, scale=0.1`
  dans `train_test_partial.py`). Copier/monter les checkpoints (`config.yaml` +
  `model_checkpoint_<epoch>.eqx`) à un emplacement accessible depuis SHINE (ex.
  `data/morphology_checkpoints/`, ou chemin configurable en YAML — ne pas committer les
  poids dans git, utiliser Git LFS ou un chemin externe comme `data_path`).
- [ ] **Vérifier la cohérence des unités** : le décodeur produit directement l'image du
  profil galaxie dans les mêmes unités (ADU/count) que les `sci_subtracted` utilisés à
  l'entraînement. Confirmer que ce sont les mêmes unités que les images Euclid VIS
  chargées par `shine.euclid.data_loader` (pas de facteur de calibration photométrique
  supplémentaire caché entre les deux jeux de données).

---

## Étape 1 — Porter le code du modèle dans `shine/morphology/`

Créer un nouveau sous-package `shine/morphology/` (module prévu par `DESIGN.md` §3.5,
actuellement "Planned"). Ne pas dépendre du repo `Train-AE` directement (pas un package
installable propre) : **porter** les fichiers nécessaires.

- [ ] `shine/morphology/__init__.py`
- [ ] `shine/morphology/nn/` — copier depuis `Train-AE/pshear/nn/` : `blocks.py`,
  `layers.py`, `utils.py`, `autoencoder.py`, `flow.py` (adapter les imports relatifs).
- [ ] `shine/morphology/galaxy_ae.py` — copier `GalaxyAutoEncoder`,
  `make_galaxy_autoencoder` depuis `pshear/galaxy.py`. **Ne pas** copier
  `GalaxyAutoEncoderLoss` (utile seulement à l'entraînement, hors périmètre de SHINE).
- [ ] `shine/morphology/loader.py` — adapter `load_galaxy_autoencoder()` / `load_flow()`
  de `pshear/utils.py`, avec une différence importante :
  ```python
  import equinox as eqx

  def load_frozen_autoencoder(model_path, epoch):
      ae = load_galaxy_autoencoder(model_path, epoch)
      # Obligatoire : désactive tout Dropout -> rend le modèle purement déterministe
      ae = eqx.nn.inference_mode(ae, value=True)
      return ae
  ```
  Idem pour le flow si celui-ci contient du dropout (vérifier `nn_width`/`nn_depth` MLP
  internes de flowjax — a priori pas de dropout côté flow, mais à vérifier).
- [ ] Charger l'AE et le flow **une seule fois** (au moment de la construction de
  `MultiExposureScene`, pas à chaque appel du modèle NumPyro) et les garder comme
  attributs Python normaux (closure), jamais comme site `numpyro.sample` ni paramètre
  optimisé par SVI/NUTS.

---

## Étape 2 — Nouveau bloc de config

Ajouter dans `shine/morphology/config.py` un bloc Pydantic dédié, et le brancher dans
`EuclidInferenceConfig` (`shine/euclid/config.py`) :

```python
# shine/morphology/config.py
from pydantic import BaseModel

class LearnedMorphologyConfig(BaseModel):
    enabled: bool = False
    ae_checkpoint_dir: str
    ae_epoch: int
    flow_checkpoint_dir: str
    flow_epoch: int
    apply_to_stamp_size: int = 64   # doit correspondre à un élément de galaxy_stamp_sizes
```

```python
# shine/euclid/config.py — ajout d'un champ optionnel
from shine.morphology.config import LearnedMorphologyConfig

class EuclidInferenceConfig(BaseModel):
    ...
    learned_morphology: Optional[LearnedMorphologyConfig] = None
```

- [ ] Valider (`field_validator`) que `apply_to_stamp_size` est bien présent dans
  `galaxy_stamp_sizes`.
- [ ] Exemple YAML (`configs/euclid_vis_learned.yaml`) dupliquant `euclid_vis.yaml` avec
  la section `learned_morphology` activée, pour ne pas casser la config par défaut.

---

## Étape 3 — Prior latent NumPyro (`shine/morphology/prior.py`)

Le flow n'est pas une `numpyro.distributions.Distribution` : on échantillonne la base
normale et on pousse à travers la transformation du flow de façon **déterministe**
(le tour standard pour un normalizing flow externe comme prior) :

```python
# shine/morphology/prior.py
import math
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def sample_latent_codes(name: str, flow, n_sources: int):
    """A appeler DANS un numpyro.plate("sources", n_sources)."""
    latent_dim_flat = math.prod(flow.latent_dim)
    z_base = numpyro.sample(
        f"{name}_base",
        dist.Normal(jnp.zeros(latent_dim_flat), 1.0).to_event(1),
    )
    z_flat = flow.forward(z_base)          # transformation déterministe (bijection)
    z = flow.unflatten_latent(z_flat)      # -> shape (latent_channels, h, w)
    numpyro.deterministic(f"{name}", z)    # pour diagnostics/traçabilité
    return z
```

- [ ] Vérifier que `flow.forward` (= `flow.flow.bijection.transform`, cf.
  `LatentFlow.forward` dans `pshear/nn/flow.py`) est bien vectorisable par
  `numpyro.plate` (batch sur la première dimension) — sinon envelopper d'un `jax.vmap`
  explicite à l'intérieur du site de sample.
- [ ] Si le flow a été entraîné **conditionnel** (`cond_dim` non nul), il faudra passer
  les covariables catalogue (ex. log flux catalogue) en argument de `flow.forward`/
  `flow.sample` — non nécessaire pour le run initial puisque le flow actuel est
  inconditionnel (`cond_dim: None` dans `train_flow.py`).

---

## Étape 4 — Rendu (`shine/morphology/render.py`)

Reproduit la logique de `render_one_galaxy` dans `shine/euclid/scene.py`, mais avec
`g = ae.decode(z_i)` à la place de `galsim.Exponential(...)`, et le **vrai** PSF/WCS par
source déjà géré par le pipeline existant (ne pas réutiliser
`GalaxyAutoEncoder.convolve()`, qui ignore le WCS Jacobien et l'offset sous-pixel).

```python
# shine/morphology/render.py
import jax_galsim as galsim

def render_learned_galaxy(
    z_i, g1, g2, psf_img, wcs_params, pix_pos, visible_i,
    ae, stamp_size, pixel_scale, gsparams,
):
    g = ae.decode(z_i, key=None)          # image "propre", pré-convolution, shape (1,64,64)
    gal_img = galsim.Image(g[0], scale=pixel_scale)
    gal = galsim.InterpolatedImage(gal_img, gsparams=gsparams)
    gal = gal.shear(g1=g1, g2=g2)          # seul paramètre physique restant

    psf = galsim.InterpolatedImage(
        galsim.Image(psf_img, scale=pixel_scale), gsparams=gsparams
    )
    final = galsim.Convolve([gal, psf], gsparams=gsparams)

    wcs = galsim.JacobianWCS(*wcs_params)
    pix_dx = pix_pos[0] / pixel_scale  # (adapter selon convention dx/dy déjà utilisée)
    stamp = final.drawImage(
        nx=stamp_size, ny=stamp_size, wcs=wcs, offset=galsim.PositionD(pix_dx, pix_pos[1]),
    ).array
    return stamp * visible_i
```

- [ ] Attention à la **direction du shear** : dans `scene.py` actuel, le shear
  intrinsèque `e1,e2` est appliqué avant le shear cosmique `g1,g2`
  (`gal.shear(e1,e2).shear(g1,g2)`). Ici il n'y a plus d'ellipticité intrinsèque
  séparée : `g` (image décodée) contient déjà la morphologie/ellipticité intrinsèque
  apprise — un seul `.shear(g1, g2)` suffit.
- [ ] `ae.decode` attend un batch `key` pour le dropout interne : passer `key=None`
  n'est valide qu'après `eqx.nn.inference_mode(ae, True)` (Étape 1). Sans ça, l'appel
  plantera ou (pire) utilisera un key implicite non désiré.

---

## Étape 5 — Brancher le tier 64 dans `MultiExposureScene`

Dans `shine/euclid/scene.py`, deux points d'accroche :

1. **`_sample_parameters()`** : au lieu d'échantillonner `flux/hlr/e1/e2` pour *toutes*
   les sources puis les indexer par tier, il faut **séparer les sources du tier appris**
   des autres *avant* l'échantillonnage :
   - sources du tier 64 → `sample_latent_codes(...)` (Étape 3), pas de flux/hlr/e1/e2 ;
   - sources des autres tiers → chemin paramétrique actuel, inchangé.
   - `g1, g2` (shear cosmique) restent globaux et partagés entre les deux chemins.
2. **`_render_tier()` / `_render_exposure_image()`** : ajouter une branche conditionnelle
   sur `tier_idx == 0` (tier 64px, si `learned_morphology.enabled`) qui appelle
   `render_learned_galaxy` (Étape 4) au lieu de `render_one_galaxy` existant. Le
   scatter-add (`jax.lax.scan` sur `all_stamps`) reste identique.

Point d'implémentation pratique : comme `flux/hlr/e1/e2/dx/dy` sont aujourd'hui des
tableaux uniques de taille `(n_sources,)` indexés par tier via `tier_indices[tier_idx]`,
il faudra soit :
- (a) **garder les mêmes tableaux** avec des valeurs "factices" (0 ou 1) pour les
  sources du tier appris (elles ne seront simplement pas utilisées dans ce chemin), et
  ajouter un tableau séparé `z` de taille `(n_sources, latent_channels, h, w)` où seules
  les lignes du tier 64 sont significatives ; ou
- (b) refactoriser légèrement `_sample_parameters()` pour retourner une structure
  (dict/NamedTuple) avec des champs optionnels par tier plutôt qu'un tuple positionnel
  fixe. **Recommandé** : (b) est plus propre mais touche à la signature interne utilisée
  par `build_model`/`build_single_exposure_model`/`render_model_images` — à faire avec
  les tests de non-régression (Étape 6) actifs.

- [ ] Garder un flag de config clair (`learned_morphology.enabled=False` par défaut) qui
  fait strictement retomber le pipeline sur le comportement actuel, testé par les 15
  tests existants (`tests/test_euclid/`).

---

## Étape 6 — Tests

- [ ] `tests/test_morphology/test_loader.py` — charge un petit checkpoint AE+flow de
  test (miniature, quelques ko), vérifie `eqx.nn.inference_mode` bien appliqué
  (dropout désactivé → sortie déterministe entre deux appels avec `key` différents).
- [ ] `tests/test_morphology/test_prior.py` — vérifie que `sample_latent_codes` produit
  la bonne shape, et que la trace NumPyro (`numpyro.handlers.trace`) contient bien
  `..._base` et le déterministe `...`.
- [ ] `tests/test_morphology/test_render.py` — compare le rendu appris à une image de
  référence (chex, tolérance numérique) pour un `z` fixé.
- [ ] `tests/test_euclid/test_scene.py` — étendre avec un cas `learned_morphology.enabled
  = True` sur un jeu de sources synthétique restreint au tier 64, vérifier que
  `build_model()` compile et qu'un pas de MAP ne produit pas de NaN.
- [ ] **Non-régression** : relancer `pytest tests/test_euclid/ -q` avec
  `learned_morphology` désactivé (défaut) → doit rester 100% identique au comportement
  actuel (mêmes 15 tests verts).

---

## Étape 7 — Validation scientifique (obligatoire avant tout résultat scientifique)

- [ ] **Test d'auto-cohérence** (déjà sur la roadmap `DESIGN.md` §4.3) : générer des
  images simulées avec un shear connu en utilisant le décodeur AE (prior predictive),
  puis relancer l'inférence (NUTS/MAP) dessus et vérifier que le shear vrai retombe dans
  l'intervalle crédible.
- [ ] **Biais m/c** : mesurer un biais multiplicatif/additif sur le shear inféré, sur un
  ensemble de réalisations à shear connu variés (cf. `shine.validation`, déjà présent
  dans le repo pour ce type de test). C'est le point le plus critique : le décodeur
  appris peut introduire un biais de calibration invisible sur un seul cas, qui ne
  ressort que statistiquement sur plusieurs réalisations.
- [ ] **Sanity check du prior** : tirer des échantillons `z ~ flow.prior`, décoder, tracer
  une mosaïque (comme `sample_images` dans `train_flow.py`) et vérifier visuellement que
  le prior couvre une diversité de morphologies plausible (pas de mode collapse).

---

## Points de vigilance — résumé

| # | Risque | Mitigation |
|---|--------|-----------|
| 1 | Biais de calibration du shear caché dans le décodeur appris | Étape 7 (m/c bias) avant tout résultat scientifique |
| 2 | Dropout actif pendant l'inférence → densité log-jointe stochastique, gradients HMC invalides | `eqx.nn.inference_mode(model, value=True)` obligatoire sur AE **et** flow |
| 3 | AE/flow traités comme paramètres optimisables | Toujours les garder en closure Python (comme `self.data`), jamais en site `numpyro.sample`/`extra_args` |
| 4 | AE entraîné seulement à 64×64 px, 0.1"/px | Ne l'appliquer qu'au tier `apply_to_stamp_size=64` ; garder Sersic pour 128/256 |
| 5 | Flow inconditionnel → prior identique pour toutes les sources du tier, indépendant du flux/taille catalogue | Acceptable en V1 (prior faiblement informatif) ; envisager de ré-entraîner un flow conditionnel (`cond_dim`) si besoin d'un prior par-source plus informatif |
| 6 | Unités/flux "baked in" dans le décodeur (pas de paramètre flux séparé) | Vérifier que les unités ADU des cutouts d'entraînement correspondent à celles des quadrants VIS chargés par `data_loader.py` |
| 7 | Coût de calcul du décodeur (ResBlocks + attention) >> `galsim.Exponential` | Benchmarker sur `build_single_exposure_model` avec peu de sources avant de généraliser |
| 8 | Nouvelles dépendances (`equinox`, `flowjax`, `paramax`, `einops`) | Ajouter à `pyproject.toml`, épingler les versions utilisées par `Train-AE` |
| 9 | Poids de checkpoint volumineux | Ne pas committer directement ; Git LFS ou chemin externe configurable en YAML |

---

## Checklist condensée

- [ ] Étape 0 : dépendances + checkpoints identifiés et accessibles
- [ ] Étape 1 : `shine/morphology/` créé (nn/, galaxy_ae.py, loader.py avec inference_mode)
- [ ] Étape 2 : `LearnedMorphologyConfig` + branchement dans `EuclidInferenceConfig`
- [ ] Étape 3 : `sample_latent_codes()` dans `shine/morphology/prior.py`
- [ ] Étape 4 : `render_learned_galaxy()` dans `shine/morphology/render.py`
- [ ] Étape 5 : branche `tier_idx == 0` dans `scene.py`, flag `enabled=False` par défaut
- [ ] Étape 6 : tests unitaires + non-régression `tests/test_euclid/`
- [ ] Étape 7 : test d'auto-cohérence + biais m/c avant toute publication de résultats
