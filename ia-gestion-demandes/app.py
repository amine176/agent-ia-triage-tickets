from flask import Flask, render_template_string, request, jsonify
from classifier import classify_ticket
from similarity import trouver_ticket_similaire
from summary import generer_resume

app = Flask(__name__)

# Tickets fictifs déjà existants, utilisés pour tester la similarité et le résumé
tickets_existants = [
    {
        "id": 1,
        "titre": "PC ne démarre plus",
        "description": "Écran noir depuis ce matin, impossible de travailler",
        "categorie": "Incident IT",
        "departement": "DSI",
        "gravite": "Moyenne",
        "statut": "Ouvert"
    },
    {
        "id": 2,
        "titre": "Serveur mail en panne",
        "description": "Impossible d'envoyer ou recevoir des emails depuis ce matin",
        "categorie": "Incident IT",
        "departement": "DSI",
        "gravite": "Critique",
        "statut": "Ouvert"
    },
    {
        "id": 3,
        "titre": "Demande de congé",
        "description": "Je souhaite poser une semaine de congé en août",
        "categorie": "Demande RH",
        "departement": "RH",
        "gravite": "Faible",
        "statut": "Fermé"
    },
    {
        "id": 4,
        "titre": "Facture impayée fournisseur",
        "description": "Un fournisseur signale une facture non réglée depuis 30 jours",
        "categorie": "Demande Finance",
        "departement": "Finance",
        "gravite": "Élevée",
        "statut": "Ouvert"
    },
]

PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent IA — Triage des demandes</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#0E141B;
    --panel:#161F2A;
    --panel-2:#1C2733;
    --line:#28323F;
    --text:#E7ECF2;
    --muted:#8B96A5;
    --amber:#E8A33D;
    --cyan:#4FB6C7;
    --red:#D6544B;
    --green:#6FCF97;
  }
  *{box-sizing:border-box;}
  body{
    margin:0;
    background:var(--ink);
    color:var(--text);
    font-family:'Inter',sans-serif;
    min-height:100vh;
    display:flex;
    flex-direction:column;
  }
  header{
    padding:22px 28px 16px;
    border-bottom:1px solid var(--line);
  }
  .eyebrow{
    font-family:'JetBrains Mono',monospace;
    font-size:11px;
    letter-spacing:.14em;
    color:var(--cyan);
    text-transform:uppercase;
  }
  h1{
    margin:6px 0 0;
    font-size:20px;
    font-weight:600;
  }
  main{
    flex:1;
    max-width:760px;
    width:100%;
    margin:0 auto;
    padding:24px 20px 140px;
  }
  #log{
    display:flex;
    flex-direction:column;
    gap:18px;
  }
  .empty{
    color:var(--muted);
    font-size:14px;
    padding:40px 0;
    text-align:center;
    border:1px dashed var(--line);
    border-radius:10px;
  }
  .msg-user{
    align-self:flex-end;
    max-width:78%;
    background:var(--panel-2);
    border:1px solid var(--line);
    border-radius:14px 14px 4px 14px;
    padding:12px 16px;
    font-size:14px;
    line-height:1.5;
    position:relative;
  }
  .msg-user::after{
    content:"";
    position:absolute;
    right:-1px; top:0; bottom:0;
    width:6px;
    background:repeating-linear-gradient(180deg, transparent 0 6px, var(--ink) 6px 8px);
  }
  .card{
    align-self:flex-start;
    max-width:92%;
    width:100%;
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:4px 14px 14px 14px;
    overflow:hidden;
  }
  .card-top{
    display:flex;
    flex-wrap:wrap;
    gap:6px;
    padding:12px 16px 10px;
    border-bottom:1px solid var(--line);
  }
  .tag{
    font-family:'JetBrains Mono',monospace;
    font-size:11px;
    padding:3px 8px;
    border-radius:20px;
    border:1px solid var(--line);
    color:var(--muted);
    white-space:nowrap;
  }
  .tag.dept{ color:var(--cyan); border-color:rgba(79,182,199,.35); }
  .tag.prio-haute, .tag.prio-urgente{ color:var(--amber); border-color:rgba(232,163,61,.4); }
  .tag.grav-critique, .tag.grav-elevee{ color:var(--red); border-color:rgba(214,84,75,.4); }
  .tag.grav-faible, .tag.prio-basse{ color:var(--green); border-color:rgba(111,207,151,.35); }
  .card-body{
    padding:14px 16px 16px;
    font-size:14px;
    line-height:1.6;
  }
  .card-body p{ margin:0 0 8px; }
  .link-ticket{
    margin-top:6px;
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    color:var(--cyan);
    display:flex;
    align-items:center;
    gap:6px;
  }
  .scan-card{
    align-self:flex-start;
    width:70%;
    height:64px;
    border-radius:4px 14px 14px 14px;
    border:1px solid var(--line);
    background:var(--panel);
    position:relative;
    overflow:hidden;
  }
  .scan-card::before{
    content:"";
    position:absolute;
    top:0; left:-40%;
    width:40%; height:100%;
    background:linear-gradient(90deg, transparent, rgba(79,182,199,.35), transparent);
    animation:scan 1.1s linear infinite;
  }
  @keyframes scan{
    to{ left:140%; }
  }
  footer{
    position:fixed;
    bottom:0; left:0; right:0;
    background:linear-gradient(0deg, var(--ink) 60%, transparent);
    padding:22px 20px 24px;
  }
  .composer{
    max-width:760px;
    margin:0 auto;
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:14px;
    padding:10px 10px 10px 16px;
    display:flex;
    gap:10px;
    align-items:flex-end;
  }
  .composer select{
    background:var(--panel-2);
    color:var(--muted);
    border:1px solid var(--line);
    border-radius:8px;
    font-family:'JetBrains Mono',monospace;
    font-size:11px;
    padding:6px 8px;
  }
  .composer textarea{
    flex:1;
    background:transparent;
    border:none;
    color:var(--text);
    font-family:'Inter',sans-serif;
    font-size:14px;
    resize:none;
    outline:none;
    padding:8px 0;
    max-height:120px;
  }
  .composer textarea::placeholder{ color:var(--muted); }
  .composer button{
    background:var(--cyan);
    color:var(--ink);
    border:none;
    border-radius:10px;
    padding:10px 16px;
    font-weight:600;
    font-size:13px;
    cursor:pointer;
    white-space:nowrap;
  }
  .composer button:disabled{ opacity:.5; cursor:default; }
  .composer button.ghost{
    background:transparent;
    border:1px solid var(--line);
    color:var(--muted);
  }
</style>
</head>
<body>

<header>
  <div class="eyebrow">Agent IA · Mistral</div>
  <h1>Console de triage des demandes</h1>
</header>

<main>
  <div id="log">
    <div class="empty" id="empty-state">Décrivez un incident ou une réclamation ci-dessous pour lancer l'analyse.</div>
  </div>
</main>

<footer>
  <div class="composer">
    <select id="poste">
      <option value="Collaborateur">Collaborateur</option>
      <option value="Manager">Manager</option>
      <option value="Chef de département">Chef dépt.</option>
      <option value="Directeur">Directeur</option>
    </select>
    <textarea id="input" rows="1" placeholder="Ex : le serveur mail est down depuis ce matin..."></textarea>
    <button class="ghost" id="btn-resume" type="button">Résumé semaine</button>
    <button id="btn-send" type="button">Analyser</button>
  </div>
</footer>

<script>
const log = document.getElementById('log');
const input = document.getElementById('input');
const posteSelect = document.getElementById('poste');
const btnSend = document.getElementById('btn-send');
const btnResume = document.getElementById('btn-resume');
const emptyState = document.getElementById('empty-state');

function hideEmptyState(){
  if(emptyState) emptyState.remove();
}

function addUserBubble(text){
  hideEmptyState();
  const div = document.createElement('div');
  div.className = 'msg-user';
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function addScanCard(){
  hideEmptyState();
  const div = document.createElement('div');
  div.className = 'scan-card';
  div.id = 'scan-current';
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

function slug(str){
  return (str || '').toLowerCase()
    .normalize('NFD').replace(/[\\u0300-\\u036f]/g, '')
    .replace(/\\s+/g, '-');
}

function renderClassificationCard(result){
  const scan = document.getElementById('scan-current');
  if(scan) scan.remove();

  const c = result.classification;
  const s = result.similarite;

  const card = document.createElement('div');
  card.className = 'card';

  const top = document.createElement('div');
  top.className = 'card-top';
  top.innerHTML = `
    <span class="tag dept">${c.departement}</span>
    <span class="tag grav-${slug(c.gravite)}">${c.gravite}</span>
    <span class="tag prio-${slug(c.priorite)}">${c.priorite}</span>
    <span class="tag">${c.categorie}</span>
    ${c.impact_poste ? `<span class="tag">impact poste: ${c.impact_poste}</span>` : ''}
  `;

  const body = document.createElement('div');
  body.className = 'card-body';
  let html = `<p>Ticket classé automatiquement par l'agent Mistral.</p>`;
  if(s.trouve){
    html += `<div class="link-ticket">🔗 Correspond au ticket existant #${s.ticket_id} — rattaché au même groupe de problème.</div>`;
  } else {
    html += `<div class="link-ticket" style="color:var(--muted)">— Aucun ticket similaire ouvert détecté, nouveau problème.</div>`;
  }
  body.innerHTML = html;

  card.appendChild(top);
  card.appendChild(body);
  log.appendChild(card);
  log.scrollTop = log.scrollHeight;
}

function mdToHtml(texte){
  // Échappement HTML de base
  let html = texte
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Titres (### avant ## avant #, pour matcher le plus spécifique en premier)
  html = html.replace(/^### (.*)$/gm, '<h4>$1</h4>');
  html = html.replace(/^## (.*)$/gm, '<h3>$1</h3>');
  html = html.replace(/^# (.*)$/gm, '<h2>$1</h2>');

  // Gras et italique
  html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  html = html.replace(/(^|[^*])\\*(?!\\*)(.+?)\\*(?!\\*)/g, '$1<em>$2</em>');

  // Listes à puces : on regroupe les lignes "- x" ou "* x" consécutives en <ul>
  html = html.replace(/(?:^|\\n)((?:[-*] .*(?:\\n|$))+)/g, (match, block) => {
    const items = block.trim().split('\\n')
      .map(l => l.replace(/^[-*]\\s+/, '').trim())
      .map(l => `<li>${l}</li>`).join('');
    return `\\n<ul>${items}</ul>\\n`;
  });

  // Listes numérotées : "1. x", "2. x" ...
  html = html.replace(/(?:^|\\n)((?:\\d+\\. .*(?:\\n|$))+)/g, (match, block) => {
    const items = block.trim().split('\\n')
      .map(l => l.replace(/^\\d+\\.\\s+/, '').trim())
      .map(l => `<li>${l}</li>`).join('');
    return `\\n<ol>${items}</ol>\\n`;
  });

  // Retours à la ligne restants (hors listes/titres déjà englobés dans des balises)
  html = html.replace(/\\n{2,}/g, '</p><p>');
  html = html.replace(/\\n/g, '<br>');
  html = `<p>${html}</p>`;

  // Nettoyage : éviter des <p> vides autour des <ul>/<ol>/<h*>
  html = html.replace(/<p>(<h\\d>|<ul>|<ol>)/g, '$1');
  html = html.replace(/(<\\/h\\d>|<\\/ul>|<\\/ol>)<\\/p>/g, '$1');

  return html;
}

function renderResumeCard(texte){
  const scan = document.getElementById('scan-current');
  if(scan) scan.remove();

  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `
    <div class="card-top"><span class="tag">résumé hebdomadaire</span></div>
    <div class="card-body">${mdToHtml(texte)}</div>
  `;
  log.appendChild(card);
  log.scrollTop = log.scrollHeight;
}

async function envoyer(){
  const texte = input.value.trim();
  if(!texte) return;

  addUserBubble(texte);
  input.value = '';
  btnSend.disabled = true;
  addScanCard();

  try{
    const res = await fetch('/api/analyser', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ message: texte, poste: posteSelect.value })
    });
    const data = await res.json();
    renderClassificationCard(data);
  } catch(e){
    const scan = document.getElementById('scan-current');
    if(scan) scan.remove();
    addUserBubble('Erreur: impossible de contacter l\\'agent IA.');
  } finally {
    btnSend.disabled = false;
  }
}

async function demanderResume(){
  hideEmptyState();
  btnResume.disabled = true;
  addScanCard();
  try{
    const res = await fetch('/api/resume');
    const data = await res.json();
    renderResumeCard(data.resume);
  } catch(e){
    const scan = document.getElementById('scan-current');
    if(scan) scan.remove();
  } finally {
    btnResume.disabled = false;
  }
}

btnSend.addEventListener('click', envoyer);
btnResume.addEventListener('click', demanderResume);
input.addEventListener('keydown', (e) => {
  if(e.key === 'Enter' && !e.shiftKey){
    e.preventDefault();
    envoyer();
  }
});
</script>

</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/analyser", methods=["POST"])
def analyser():
    data = request.get_json()
    message = data.get("message", "")
    poste = data.get("poste", "Collaborateur")

    # On utilise le premier bout de texte comme titre, tout le message comme description
    titre = message[:60]
    classification = classify_ticket(titre, message, poste)

    nouveau_ticket = {
        "titre": titre,
        "description": message,
        "categorie": classification["categorie"],
        "departement": classification["departement"]
    }
    similarite = trouver_ticket_similaire(nouveau_ticket, tickets_existants)

    return jsonify({
        "classification": classification,
        "similarite": similarite
    })


@app.route("/api/resume")
def resume():
    texte = generer_resume(tickets_existants)
    return jsonify({"resume": texte})


if __name__ == "__main__":
    app.run(debug=True)
