"""Dataset builder for Inspector AI.

Generates an expanded mock training dataset of text pairs used to teach Inspector
critical thinking from absolute zero:

1. Vague / dubious prompts  ->  strategic clarification questions
   (teaches the model to recognize ambiguity and ask instead of guessing)

2. Clear prompts            ->  R.C.F.O formatted prompts
   (teaches the model to decompose a clear request into Role, Context,
    Focus, and Output)

The output is written to inspector/data/raw/mock_dataset.json
and inspector/data/processed/mock_dataset.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "inspector" / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "inspector" / "data" / "processed"

# ---------------------------------------------------------------------------
# Part 1: Vague / dubious prompts -> strategic clarification questions
# Each pair teaches Inspector to detect under-specification and respond
# with strategic, R.C.F.O-aligned questions instead of hallucinating.
# ---------------------------------------------------------------------------
VAGUE_TO_QUESTIONS: List[Tuple[str, str]] = [
    # --- Technology / Software ------------------------------------------------
    ("fix my code", "Role: Who should fix the code?\nContext: What code and what problem?\nFocus: Which specific bug or issue?\nOutput: Fixed code or bug report?"),
    ("fix my website", "Role: Who should fix the website?\nContext: What website and what is wrong?\nFocus: Which specific issue or goal?\nOutput: Bug report, patched code, or live fix?"),
    ("debug this", "Role: Who is debugging?\nContext: What is 'this' and what error is observed?\nFocus: Which part or symptom first?\nOutput: Root cause, fix, test case, or docs?"),
    ("debug my app", "Role: Who debugs the app?\nContext: What app and platform?\nFocus: Which crash or bug?\nOutput: Root cause or fix?"),
    ("make it work", "Role: Who is responsible?\nContext: What is 'it' and what is broken?\nFocus: Which part should be fixed?\nOutput: Working solution or docs?"),
    ("make this better", "Role: Who improves?\nContext: What is 'this' and its shortcomings?\nFocus: Performance, clarity, design, or accuracy?\nOutput: What does 'better' look like?"),
    ("improve the design", "Role: Who improves?\nContext: What design and shortcomings?\nFocus: Usability, visual, or interaction?\nOutput: Wireframe, mockup, or spec?"),
    ("improve performance", "Role: Who optimizes?\nContext: What system or process?\nFocus: Speed, latency, or throughput?\nOutput: Benchmark report or optimized code?"),
    ("analyze the data", "Role: Who analyzes?\nContext: What data and where?\nFocus: Which questions or metrics?\nOutput: Chart, report, or summary?"),
    ("analyze this data", "Role: Who performs analysis?\nContext: What dataset?\nFocus: What insights or metrics?\nOutput: Dashboard, report, or CSV?"),
    ("create a presentation", "Role: Who creates and presents?\nContext: What topic and audience?\nFocus: Key messages or objectives?\nOutput: Slide deck, speaker notes, handout?"),
    ("build a website", "Role: Who builds?\nContext: What kind of site?\nFocus: Which features or pages?\nOutput: Live site, prototype, or code?"),
    ("build an app", "Role: Who builds?\nContext: What problem does it solve?\nFocus: Core features or MVP?\nOutput: Wireframe, doc, or prototype?"),
    ("write something", "Role: Who writes?\nContext: What topic?\nFocus: What specific piece?\nOutput: Blog post, ad copy, email, or deck?"),
    ("write an email", "Role: Sender and recipient?\nContext: What purpose?\nFocus: Key points or actions?\nOutput: Formal, casual, request, or announcement?"),
    ("send a message", "Role: Sender and recipient?\nContext: What channel and purpose?\nFocus: What to communicate?\nOutput: Text, Slack, email, or chat?"),
    ("help me", "Role: Who helps?\nContext: What situation?\nFocus: What specific assistance?\nOutput: What kind of help or deliverable?"),
    ("help me with this", "Role: Who helps?\nContext: What is 'this'?\nFocus: What part needs help?\nOutput: Code review, planning, writing, debugging?"),
    ("I need help", "Role: Who assists?\nContext: What project or task?\nFocus: Which part?\nOutput: Guidance, code, or document?"),
    ("what should I do?", "Role: Who decides?\nContext: What situation or goal?\nFocus: Which decision or action?\nOutput: Recommendation, plan, or steps?"),
    ("what do I do now?", "Role: Who decides next?\nContext: What happened?\nFocus: Next step or recovery?\nOutput: Action plan or instructions?"),
    ("i need a report", "Role: Who writes?\nContext: Report on what topic?\nFocus: Which sections or metrics?\nOutput: PDF, Word, or slides?"),
    ("create a report", "Role: Who creates?\nContext: What subject?\nFocus: Which data or findings?\nOutput: Document, deck, or dashboard?"),
    ("marketing strategy", "Role: Who creates?\nContext: What product or market?\nFocus: Which channels or tactics?\nOutput: Document, deck, or plan?"),
    ("marketing campaign", "Role: Who runs it?\nContext: What product and audience?\nFocus: Which channels or goals?\nOutput: Plan, assets, or results?"),
    ("sales pitch", "Role: Who presents?\nContext: What product and prospect?\nFocus: Which pain point or value?\nOutput: Deck, script, or one-pager?"),
    ("customer feedback", "Role: Who collects?\nContext: Feedback about what?\nFocus: Which customers or touchpoints?\nOutput: Survey, summary, or action plan?"),
    ("project plan", "Role: Who plans?\nContext: What project and timeline?\nFocus: Which phases or milestones?\nOutput: Gantt, tasks, or document?"),
    ("project status", "Role: Who reports?\nContext: What project?\nFocus: Progress, blockers, or next steps?\nOutput: Status email, deck, or doc?"),
    ("technical documentation", "Role: Who writes?\nContext: What product or system?\nFocus: API, user guide, or architecture?\nOutput: Markdown, HTML, or PDF?"),
    ("API documentation", "Role: Who writes?\nContext: What API?\nFocus: Endpoints, auth, or examples?\nOutput: Markdown, Swagger, or HTML?"),
    ("database schema", "Role: Who designs?\nContext: What application?\nFocus: Tables, relationships, or constraints?\nOutput: SQL, diagram, or doc?"),
    ("security audit", "Role: Who audits?\nContext: What system or network?\nFocus: Vulnerabilities or compliance?\nOutput: Report, findings, or remediation plan?"),
    ("performance review", "Role: Who conducts?\nContext: What employee and period?\nFocus: Goals, achievements, or growth?\nOutput: Form, document, or meeting?"),
    ("onboarding process", "Role: Who designs?\nContext: What role or company?\nFocus: First week, month, or quarter?\nOutput: Checklist, document, or deck?"),
    ("exit interview", "Role: Who conducts?\nContext: What employee?\nFocus: Reasons for leaving or feedback?\nOutput: Form, document, or summary?"),
    ("survey design", "Role: Who designs?\nContext: What topic or audience?\nFocus: Question types or scale?\nOutput: Form, document, or link?"),
    ("interview questions", "Role: Who writes?\nContext: What role?\nFocus: Behavioral, technical, or situational?\nOutput: List, document, or scorecard?"),
    ("hiring plan", "Role: Who creates?\nContext: What roles and timeline?\nFocus: Sourcing, screening, or offers?\nOutput: Document, deck, or spreadsheet?"),
    ("training program", "Role: Who designs?\nContext: What skills or audience?\nFocus: Curriculum, delivery, or assessment?\nOutput: Syllabus, deck, or LMS?"),
    ("workshop agenda", "Role: Who facilitates?\nContext: What topic and duration?\nFocus: Activities, breaks, or outcomes?\nOutput: Document, slides, or board?"),
    ("conference talk", "Role: Who speaks?\nContext: What event and audience?\nFocus: Key message or takeaway?\nOutput: Slides, script, or handout?"),
    ("webinar script", "Role: Who presents?\nContext: What topic and platform?\nFocus: Introduction, content, or Q&A?\nOutput: Script, slides, or runbook?"),
    ("podcast outline", "Role: Who hosts?\nContext: What topic and guest?\nFocus: Segments, questions, or stories?\nOutput: Document, script, or notes?"),
    ("video script", "Role: Who writes?\nContext: What format and length?\nFocus: Hook, body, or CTA?\nOutput: Script, storyboard, or doc?"),
    ("advertisement", "Role: Who creates?\nContext: What product and channel?\nFocus: Headline, copy, or visual?\nOutput: Copy, mockup, or campaign?"),
    ("product description", "Role: Who writes?\nContext: What product?\nFocus: Features, benefits, or specs?\nOutput: Webpage, PDF, or catalog?"),
    ("pricing page", "Role: Who designs?\nContext: What product or tiers?\nFocus: Plans, features, or CTA?\nOutput: Webpage, Figma, or doc?"),
    ("terms of service", "Role: Who drafts?\nContext: What platform?\nFocus: User conduct, liability, or termination?\nOutput: PDF, webpage, or legal doc?"),
    ("privacy policy", "Role: Who drafts?\nContext: What data or region?\nFocus: Collection, use, or sharing?\nOutput: PDF, webpage, or legal doc?"),
    ("cookie policy", "Role: Who drafts?\nContext: What website?\nFocus: Types, purpose, or consent?\nOutput: PDF, webpage, or banner?"),
    ("refund policy", "Role: Who writes?\nContext: What product or region?\nFocus: Eligibility, process, or timeline?\nOutput: PDF, webpage, or FAQ?"),
    ("shipping policy", "Role: Who writes?\nContext: What store or region?\nFocus: Methods, costs, or delivery?\nOutput: PDF, webpage, or FAQ?"),
    ("about page", "Role: Who writes?\nContext: What company?\nFocus: History, mission, or team?\nOutput: Webpage, PDF, or deck?"),
    ("contact page", "Role: Who writes?\nContext: What company?\nFocus: Form, email, or phone?\nOutput: Webpage, form, or doc?"),
    ("thank you page", "Role: Who writes?\nContext: What action?\nFocus: Gratitude, next steps, or CTA?\nOutput: Webpage, email, or pop-up?"),
    ("landing page", "Role: Who builds?\nContext: What offer or campaign?\nFocus: Headline, copy, or CTA?\nOutput: HTML, Figma, or live page?"),
    ("pitch deck", "Role: Who presents?\nContext: What startup or idea?\nFocus: Problem, solution, or traction?\nOutput: Slides, PDF, or video?"),
    ("investor update", "Role: Who sends?\nContext: What company and period?\nFocus: Metrics, milestones, or asks?\nOutput: Email, deck, or document?"),
    ("board meeting", "Role: Who organizes?\nContext: What organization?\nFocus: Agenda, minutes, or decisions?\nOutput: Document, slides, or minutes?"),
    ("team retrospective", "Role: Who facilitates?\nContext: What sprint or project?\nFocus: What went well or poorly?\nOutput: Document, board, or action items?"),
    ("sprint planning", "Role: Who leads?\nContext: What team and backlog?\nFocus: Goals, tasks, or capacity?\nOutput: Board, tasks, or document?"),
    ("bug report", "Role: Who reports?\nContext: What software or version?\nFocus: Steps to reproduce or impact?\nOutput: Ticket, document, or email?"),
    ("feature request", "Role: Who requests?\nContext: What product?\nFocus: User need or benefit?\nOutput: Ticket, doc, or spec?"),
    ("user guide", "Role: Who writes?\nContext: What product?\nFocus: Setup, usage, or troubleshooting?\nOutput: PDF, HTML, or help center?"),
    ("training manual", "Role: Who writes?\nContext: What system or process?\nFocus: Onboarding or skills?\nOutput: PDF, video, or interactive?"),
    ("policy document", "Role: Who writes?\nContext: What policy area?\nFocus: Rules, procedures, or compliance?\nOutput: PDF, Word, or intranet page?"),
    ("press release", "Role: Who issues?\nContext: What news?\nFocus: Launch, milestone, or crisis?\nOutput: PDF, Word, or email?"),
    ("job description", "Role: Who writes?\nContext: What role and team?\nFocus: Responsibilities or requirements?\nOutput: PDF, webpage, or ATS?"),
    ("resume", "Role: Who writes?\nContext: What industry or level?\nFocus: Experience or skills?\nOutput: PDF, Word, or LinkedIn?"),
    ("cover letter", "Role: Who writes?\nContext: What job and company?\nFocus: Motivation or fit?\nOutput: PDF, email, or form?"),
    ("LinkedIn post", "Role: Who posts?\nContext: What topic?\nFocus: Thought leadership or hiring?\nOutput: Text, image, or article?"),
    ("blog article", "Role: Who writes?\nContext: What topic and audience?\nFocus: Educational, opinion, or news?\nOutput: WordPress, Medium, or PDF?"),
    ("newsletter", "Role: Who sends?\nContext: What audience and frequency?\nFocus: Curated content or updates?\nOutput: HTML, email, or PDF?"),
    ("whitepaper", "Role: Who writes?\nContext: What technology?\nFocus: Problem, solution, or evidence?\nOutput: PDF, landing page, or deck?"),
    ("case study", "Role: Who writes?\nContext: What client or project?\nFocus: Challenge, solution, or results?\nOutput: PDF, webpage, or slide?"),
    ("testimonial", "Role: Who gives?\nContext: What product or service?\nFocus: Experience or results?\nOutput: Text, video, or quote?"),
    ("FAQ page", "Role: Who writes?\nContext: What product?\nFocus: Common questions or issues?\nOutput: Webpage, PDF, or help center?"),
    ("financial analysis", "Role: Who analyzes?\nContext: What company or period?\nFocus: Revenue, profit, or cash flow?\nOutput: Spreadsheet, report, or deck?"),
    ("budget proposal", "Role: Who submits?\nContext: What initiative?\nFocus: Which costs or categories?\nOutput: Spreadsheet, document, or deck?"),
    ("research paper", "Role: Who writes?\nContext: What field or question?\nFocus: Literature review or experiment?\nOutput: PDF, LaTeX, or Word?"),
    ("social media post", "Role: Who posts?\nContext: What platform?\nFocus: Announcement, engagement, or promotion?\nOutput: Text, image, or video?"),
    ("startup plan", "Role: Who founds?\nContext: What market?\nFocus: Go-to-market or fundraising?\nOutput: Deck, document, or financial model?"),
    ("legal document", "Role: Who drafts?\nContext: What agreement?\nFocus: Parties, terms, or obligations?\nOutput: PDF, Word, or template?"),
    ("medical report", "Role: Who writes?\nContext: What patient?\nFocus: Diagnosis, treatment, or prognosis?\nOutput: PDF, chart, or summary?"),
    ("app idea", "Role: Who builds?\nContext: What problem?\nFocus: Core features or MVP?\nOutput: Wireframe, doc, or prototype?"),
    ("create a system", "Role: Who builds?\nContext: What system and purpose?\nFocus: Which components or scope?\nOutput: Architecture, code, or docs?"),
    ("create a process", "Role: Who designs?\nContext: What process?\nFocus: Steps, owners, or tools?\nOutput: Flowchart, doc, or SOP?"),
    ("create a template", "Role: Who creates?\nContext: What kind of document?\nFocus: Sections or fields?\nOutput: File, example, or guide?"),
    ("write a proposal", "Role: Who writes?\nContext: What project or client?\nFocus: Scope, timeline, or price?\nOutput: Document, deck, or PDF?"),
    ("write a spec", "Role: Who writes?\nContext: What feature?\nFocus: Requirements or acceptance criteria?\nOutput: Document, doc, or ticket?"),
    ("write a brief", "Role: Who writes?\nContext: What campaign?\nFocus: Objectives, audience, or message?\nOutput: Document, deck, or email?"),
    ("write a script", "Role: Who writes?\nContext: What video or call?\nFocus: Dialogue or structure?\nOutput: Script, doc, or video?"),
    ("write a summary", "Role: Who writes?\nContext: What source?\nFocus: Key points or conclusions?\nOutput: Document, email, or slide?"),
    ("write a review", "Role: Who writes?\nContext: What product or service?\nFocus: Pros, cons, or rating?\nOutput: Text, review, or post?"),
    ("write a bio", "Role: Who writes?\nContext: What person?\nFocus: Experience or achievements?\nOutput: Text, webpage, or PDF?"),
    ("write a statement", "Role: Who writes?\nContext: What situation?\nFocus: Position, apology, or clarification?\nOutput: Text, email, or letter?"),
    ("create a dashboard", "Role: Who builds?\nContext: What metrics?\nFocus: Which charts or KPIs?\nOutput: Tool, link, or screenshot?"),
    ("create a chart", "Role: Who creates?\nContext: What data?\nFocus: Which type or insight?\nOutput: Image, file, or link?"),
    ("create a form", "Role: Who builds?\nContext: What purpose?\nFocus: Which fields or validations?\nOutput: Link, file, or embedded form?"),
    ("create a checklist", "Role: Who creates?\nContext: What process?\nFocus: Which steps or items?\nOutput: Document, list, or board?"),
    ("create a timeline", "Role: Who creates?\nContext: What project?\nFocus: Which dates or milestones?\nOutput: Gantt, calendar, or doc?"),
    ("create a roadmap", "Role: Who creates?\nContext: What product?\nFocus: Which features or quarters?\nOutput: Document, board, or deck?"),
    ("create a strategy", "Role: Who creates?\nContext: What area?\nFocus: Goals, tactics, or KPIs?\nOutput: Document, deck, or plan?"),
    ("create a framework", "Role: Who creates?\nContext: What domain?\nFocus: Components or principles?\nOutput: Document, diagram, or deck?"),
    ("create a model", "Role: Who builds?\nContext: What problem?\nFocus: Which algorithm or data?\nOutput: Code, notebook, or model file?"),
    ("create a prototype", "Role: Who builds?\nContext: What idea?\nFocus: Which features or flow?\nOutput: Figma, code, or mockup?"),
    ("create a mockup", "Role: Who designs?\nContext: What screen?\nFocus: Layout or visual?\nOutput: Figma, image, or file?"),
    ("create a wireframe", "Role: Who designs?\nContext: What page?\nFocus: Structure or elements?\nOutput: Figma, file, or image?"),
    ("review this", "Role: Who reviews?\nContext: What artifact?\nFocus: Quality, correctness, or compliance?\nOutput: Feedback, doc, or approval?"),
    ("approve this", "Role: Who approves?\nContext: What request?\nFocus: Criteria or conditions?\nOutput: Approval, email, or signature?"),
    ("sign this", "Role: Who signs?\nContext: What document?\nFocus: Authority or consent?\nOutput: Signed document or confirmation?"),
    ("send this", "Role: Who sends?\nContext: What and to whom?\nFocus: Channel or timing?\nOutput: Confirmation or receipt?"),
    ("publish this", "Role: Who publishes?\nContext: What content and where?\nFocus: Timing or audience?\nOutput: Link or confirmation?"),
    ("launch this", "Role: Who launches?\nContext: What product?\nFocus: Date or channels?\nOutput: Launch plan or announcement?"),
    ("cancel this", "Role: Who cancels?\nContext: What event or order?\nFocus: Reason or next steps?\nOutput: Confirmation or notice?"),
    ("update this", "Role: Who updates?\nContext: What artifact?\nFocus: Which fields or version?\nOutput: Updated file or confirmation?"),
    ("delete this", "Role: Who deletes?\nContext: What item?\nFocus: Confirmation or backup?\nOutput: Confirmation or log?"),
    ("move this", "Role: Who moves?\nContext: What item and where?\nFocus: Permissions or impact?\nOutput: Confirmation or new location?"),
    ("copy this", "Role: Who copies?\nContext: What item and where?\nFocus: Duplicate or link?\nOutput: Confirmation or new copy?"),
    ("share this", "Role: Who shares?\nContext: What and with whom?\nFocus: Access level or channel?\nOutput: Link or confirmation?"),
    ("export this", "Role: Who exports?\nContext: What data?\nFocus: Format or destination?\nOutput: File or download link?"),
    ("import this", "Role: Who imports?\nContext: What file?\nFocus: Mapping or validation?\nOutput: Confirmation or errors?"),
    ("convert this", "Role: Who converts?\nContext: What file?\nFocus: Target format or settings?\nOutput: Converted file or link?"),
    ("compress this", "Role: Who compresses?\nContext: What file?\nFocus: Format or level?\nOutput: Compressed file or link?"),
    ("encrypt this", "Role: Who encrypts?\nContext: What data?\nFocus: Algorithm or key?\nOutput: Encrypted file or key?"),
    ("decrypt this", "Role: Who decrypts?\nContext: What data?\nFocus: Key or method?\nOutput: Decrypted data or file?"),
    ("translate this", "Role: Who translates?\nContext: What text and languages?\nFocus: Tone or audience?\nOutput: Translated text or file?"),
    ("transcribe this", "Role: Who transcribes?\nContext: What audio?\nFocus: Verbatim or summary?\nOutput: Text file or document?"),
    ("summarize this", "Role: Who summarizes?\nContext: What text?\nFocus: Key points or length?\nOutput: Summary or bullet list?"),
    ("expand this", "Role: Who expands?\nContext: What text?\nFocus: More detail or examples?\nOutput: Expanded text or document?"),
    ("rewrite this", "Role: Who rewrites?\nContext: What text?\nFocus: Tone, clarity, or audience?\nOutput: Rewritten text or document?"),
    ("proofread this", "Role: Who proofreads?\nContext: What text?\nFocus: Grammar, spelling, or style?\nOutput: Corrected text or comments?"),
    ("format this", "Role: Who formats?\nContext: What document?\nFocus: Style, layout, or template?\nOutput: Formatted file or link?"),
    ("design this", "Role: Who designs?\nContext: What artifact?\nFocus: Visuals, layout, or brand?\nOutput: Mockup, file, or spec?"),
    ("illustrate this", "Role: Who illustrates?\nContext: What concept?\nFocus: Style or medium?\nOutput: Image, file, or link?"),
    ("animate this", "Role: Who animates?\nContext: What asset?\nFocus: Style or duration?\nOutput: Video, gif, or file?"),
    ("record this", "Role: Who records?\nContext: What audio or video?\nFocus: Equipment or quality?\nOutput: File or link?"),
    ("edit this", "Role: Who edits?\nContext: What media?\nFocus: Cuts, effects, or audio?\nOutput: Edited file or link?"),
    ("mix this", "Role: Who mixes?\nContext: What audio?\nFocus: Levels, EQ, or effects?\nOutput: Mixed file or link?"),
    ("master this", "Role: Who masters?\nContext: What audio?\nFocus: Loudness or format?\nOutput: Mastered file or link?"),
    ("publish this", "Role: Who publishes?\nContext: What content?\nFocus: Platform or timing?\nOutput: Link or confirmation?"),
    ("distribute this", "Role: Who distributes?\nContext: What content?\nFocus: Channels or audience?\nOutput: Links or report?"),
    ("promote this", "Role: Who promotes?\nContext: What product?\nFocus: Channels or budget?\nOutput: Campaign or report?"),
    ("advertise this", "Role: Who advertises?\nContext: What product?\nFocus: Platform or audience?\nOutput: Campaign or results?"),
    ("sponsor this", "Role: Who sponsors?\nContext: What event?\nFocus: Budget or deliverables?\nOutput: Agreement or logo placement?"),
    ("partner with this", "Role: Who partners?\nContext: What company?\nFocus: Terms or goals?\nOutput: Agreement or MOU?"),
    ("collaborate on this", "Role: Who collaborates?\nContext: What project?\nFocus: Roles or tools?\nOutput: Plan or shared doc?"),
    ("coordinate this", "Role: Who coordinates?\nContext: What event?\nFocus: Schedule or logistics?\nOutput: Plan or calendar?"),
    ("organize this", "Role: Who organizes?\nContext: What event?\nFocus: Venue, agenda, or attendees?\nOutput: Plan or checklist?"),
    ("schedule this", "Role: Who schedules?\nContext: What meeting?\nFocus: Time, attendees, or platform?\nOutput: Calendar invite or link?"),
    ("book this", "Role: Who books?\nContext: What resource?\nFocus: Date, time, or duration?\nOutput: Confirmation or receipt?"),
    ("reserve this", "Role: Who reserves?\nContext: What item?\nFocus: Duration or purpose?\nOutput: Confirmation or receipt?"),
    ("order this", "Role: Who orders?\nContext: What item?\nFocus: Quantity or delivery?\nOutput: Order confirmation or tracking?"),
    ("buy this", "Role: Who buys?\nContext: What item?\nFocus: Budget or vendor?\nOutput: Purchase order or receipt?"),
    ("sell this", "Role: Who sells?\nContext: What item?\nFocus: Price or channel?\nOutput: Listing or sale confirmation?"),
    ("rent this", "Role: Who rents?\nContext: What item?\nFocus: Duration or cost?\nOutput: Rental agreement or confirmation?"),
    ("lease this", "Role: Who leases?\nContext: What property?\nFocus: Term or conditions?\nOutput: Lease agreement or confirmation?"),
    ("hire this", "Role: Who hires?\nContext: What role?\nFocus: Budget or start date?\nOutput: Offer letter or contract?"),
    ("fire this", "Role: Who decides?\nContext: What employee?\nFocus: Reason or process?\nOutput: Termination letter or memo?"),
    ("promote this", "Role: Who promotes?\nContext: What employee?\nFocus: New role or salary?\nOutput: Promotion letter or announcement?"),
    ("transfer this", "Role: Who transfers?\nContext: What employee?\nFocus: New team or location?\nOutput: Transfer memo or org chart?"),
    ("train this", "Role: Who trains?\nContext: What skill?\nFocus: Audience or duration?\nOutput: Training plan or materials?"),
    ("certify this", "Role: Who certifies?\nContext: What skill?\nFocus: Exam or criteria?\nOutput: Certificate or badge?"),
    ("audit this", "Role: Who audits?\nContext: What process?\nFocus: Compliance or findings?\nOutput: Audit report or findings?"),
    ("inspect this", "Role: Who inspects?\nContext: What item?\nFocus: Standards or defects?\nOutput: Inspection report or pass/fail?"),
    ("test this", "Role: Who tests?\nContext: What feature?\nFocus: Test cases or environment?\nOutput: Test results or bugs?"),
    ("validate this", "Role: Who validates?\nContext: What data?\nFocus: Rules or accuracy?\nOutput: Validation report or errors?"),
    ("verify this", "Role: Who verifies?\nContext: What claim?\nFocus: Evidence or sources?\nOutput: Verification report or confirmation?"),
    ("certify this", "Role: Who certifies?\nContext: What document?\nFocus: Authority or seal?\nOutput: Certified copy or statement?"),
    ("notarize this", "Role: Who notarizes?\nContext: What document?\nFocus: Identity or acknowledgment?\nOutput: Notarized document or seal?"),
    ("file this", "Role: Who files?\nContext: What document?\nFocus: Authority or deadline?\nOutput: Filing confirmation or receipt?"),
    ("submit this", "Role: Who submits?\nContext: What application?\nFocus: Portal or requirements?\nOutput: Submission confirmation?"),
    ("apply for this", "Role: Who applies?\nContext: What program?\nFocus: Eligibility or documents?\nOutput: Application or confirmation?"),
    ("register for this", "Role: Who registers?\nContext: What event?\nFocus: Details or payment?\nOutput: Registration confirmation?"),
    ("enroll in this", "Role: Who enrolls?\nContext: What course?\nFocus: Start date or prerequisites?\nOutput: Enrollment confirmation?"),
    ("subscribe to this", "Role: Who subscribes?\nContext: What service?\nFocus: Plan or billing?\nOutput: Subscription confirmation?"),
    ("unsubscribe from this", "Role: Who unsubscribes?\nContext: What service?\nFocus: Reason or feedback?\nOutput: Unsubscribe confirmation?"),
    ("opt in to this", "Role: Who opts in?\nContext: What program?\nFocus: Consent or data?\nOutput: Opt-in confirmation?"),
    ("opt out of this", "Role: Who opts out?\nContext: What program?\nFocus: Reason or deadline?\nOutput: Opt-out confirmation?"),
    ("join this", "Role: Who joins?\nContext: What group?\nFocus: Membership or dues?\nOutput: Membership confirmation?"),
    ("leave this", "Role: Who leaves?\nContext: What group?\nFocus: Reason or notice?\nOutput: Resignation or confirmation?"),
    ("quit this", "Role: Who quits?\nContext: What job?\nFocus: Notice or reason?\nOutput: Resignation letter or exit?"),
    ("start this", "Role: Who starts?\nContext: What project?\nFocus: Scope or timeline?\nOutput: Kickoff or charter?"),
    ("stop this", "Role: Who stops?\nContext: What activity?\nFocus: Reason or impact?\nOutput: Stop order or notice?"),
    ("pause this", "Role: Who pauses?\nContext: What process?\nFocus: Duration or trigger?\nOutput: Pause notice or status?"),
    ("resume this", "Role: Who resumes?\nContext: What process?\nFocus: When or conditions?\nOutput: Resume notice or status?"),
    ("restart this", "Role: Who restarts?\nContext: What service?\nFocus: Reason or impact?\nOutput: Restart confirmation or log?"),
    ("shutdown this", "Role: Who shuts down?\nContext: What system?\nFocus: Reason or window?\nOutput: Shutdown notice or log?"),
    ("start up this", "Role: Who starts up?\nContext: What system?\nFocus: Reason or checks?\nOutput: Startup confirmation or log?"),
    ("monitor this", "Role: Who monitors?\nContext: What metric?\nFocus: Threshold or alerts?\nOutput: Dashboard or alert?"),
    ("track this", "Role: Who tracks?\nContext: What metric?\nFocus: Frequency or method?\nOutput: Report or dashboard?"),
    ("measure this", "Role: Who measures?\nContext: What metric?\nFocus: Method or units?\nOutput: Measurement or report?"),
    ("calculate this", "Role: Who calculates?\nContext: What formula?\nFocus: Inputs or precision?\nOutput: Result or spreadsheet?"),
    ("compute this", "Role: Who computes?\nContext: What task?\nFocus: Resources or time?\nOutput: Result or log?"),
    ("estimate this", "Role: Who estimates?\nContext: What effort?\nFocus: Scope or assumptions?\nOutput: Estimate or quote?"),
    ("forecast this", "Role: Who forecasts?\nContext: What metric?\nFocus: Period or method?\nOutput: Forecast or model?"),
    ("predict this", "Role: Who predicts?\nContext: What outcome?\nFocus: Data or model?\nOutput: Prediction or probability?"),
    ("simulate this", "Role: Who simulates?\nContext: What scenario?\nFocus: Parameters or runs?\nOutput: Simulation results or report?"),
    ("model this", "Role: Who models?\nContext: What system?\nFocus: Abstraction or tool?\nOutput: Model or diagram?"),
    ("diagram this", "Role: Who diagrams?\nContext: What process?\nFocus: Notation or tool?\nOutput: Diagram or file?"),
    ("map this", "Role: Who maps?\nContext: What data?\nFocus: Fields or relationships?\nOutput: Mapping doc or file?"),
    ("chart this", "Role: Who charts?\nContext: What data?\nFocus: Type or insight?\nOutput: Chart or image?"),
    ("graph this", "Role: Who graphs?\nContext: What data?\nFocus: Type or axes?\nOutput: Graph or image?"),
    ("plot this", "Role: Who plots?\nContext: What data?\nFocus: Library or style?\nOutput: Plot or image?"),
    ("visualize this", "Role: Who visualizes?\nContext: What data?\nFocus: Tool or audience?\nOutput: Dashboard or image?"),
    ("dashboard this", "Role: Who builds?\nContext: What metrics?\nFocus: Tool or layout?\nOutput: Dashboard or link?"),
    ("report this", "Role: Who reports?\nContext: What incident?\nFocus: Severity or details?\nOutput: Incident report or ticket?"),
    ("log this", "Role: Who logs?\nContext: What event?\nFocus: Level or message?\nOutput: Log entry or file?"),
    ("track this", "Role: Who tracks?\nContext: What issue?\nFocus: Status or owner?\nOutput: Ticket or board?"),
    ("manage this", "Role: Who manages?\nContext: What resource?\nFocus: Access or lifecycle?\nOutput: Management plan or console?"),
    ("administer this", "Role: Who administers?\nContext: What system?\nFocus: Users or permissions?\nOutput: Admin console or report?"),
    ("configure this", "Role: Who configures?\nContext: What system?\nFocus: Settings or parameters?\nOutput: Config file or confirmation?"),
    ("customize this", "Role: Who customizes?\nContext: What item?\nFocus: Preferences or branding?\nOutput: Customized item or file?"),
    ("personalize this", "Role: Who personalizes?\nContext: What experience?\nFocus: Data or rules?\nOutput: Personalized item or link?"),
    ("automate this", "Role: Who automates?\nContext: What task?\nFocus: Tool or trigger?\nOutput: Automation script or workflow?"),
    ("orchestrate this", "Role: Who orchestrates?\nContext: What workflow?\nFocus: Steps or dependencies?\nOutput: Orchestration plan or file?"),
    ("schedule this", "Role: Who schedules?\nContext: What job?\nFocus: Cron or trigger?\nOutput: Schedule or job config?"),
    ("queue this", "Role: Who queues?\nContext: What task?\nFocus: Priority or retry?\nOutput: Queue status or job ID?"),
    ("deploy this", "Role: Who deploys?\nContext: What service?\nFocus: Environment or strategy?\nOutput: Deployment log or URL?"),
    ("release this", "Role: Who releases?\nContext: What version?\nFocus: Notes or rollback?\nOutput: Release notes or tag?"),
    ("rollout this", "Role: Who rolls out?\nContext: What feature?\nFocus: Percentage or regions?\nOutput: Rollout plan or status?"),
    ("rollback this", "Role: Who rolls back?\nContext: What version?\nFocus: Reason or impact?\nOutput: Rollback log or confirmation?"),
    ("hotfix this", "Role: Who hotfixes?\nContext: What bug?\nFocus: Urgency or fix?\nOutput: Patch or release?"),
    ("patch this", "Role: Who patches?\nContext: What system?\nFocus: CVE or version?\nOutput: Patch log or confirmation?"),
    ("upgrade this", "Role: Who upgrades?\nContext: What software?\nFocus: Version or compatibility?\nOutput: Upgrade log or confirmation?"),
    ("downgrade this", "Role: Who downgrades?\nContext: What software?\nFocus: Reason or version?\nOutput: Downgrade log or confirmation?"),
    ("migrate this", "Role: Who migrates?\nContext: What data?\nFocus: Source or target?\nOutput: Migration log or confirmation?"),
    ("clone this", "Role: Who clones?\nContext: What repo?\nFocus: Branch or depth?\nOutput: Clone command or path?"),
    ("fork this", "Role: Who forks?\nContext: What repo?\nFocus: Purpose or changes?\nOutput: Fork URL or repo?"),
    ("branch this", "Role: Who branches?\nContext: What repo?\nFocus: Name or base?\nOutput: Branch name or command?"),
    ("merge this", "Role: Who merges?\nContext: What branches?\nFocus: Conflicts or strategy?\nOutput: Merge commit or log?"),
    ("rebase this", "Role: Who rebases?\nContext: What branch?\nFocus: Base or conflicts?\nOutput: Rebase log or confirmation?"),
    ("commit this", "Role: Who commits?\nContext: What changes?\nFocus: Message or author?\nOutput: Commit hash or log?"),
    ("push this", "Role: Who pushes?\nContext: What branch?\nFocus: Remote or force?\nOutput: Push log or confirmation?"),
    ("pull this", "Role: Who pulls?\nContext: What branch?\nFocus: Remote or rebase?\nOutput: Pull log or confirmation?"),
    ("fetch this", "Role: Who fetches?\nContext: What remote?\nFocus: Prune or tags?\nOutput: Fetch log or confirmation?"),
    ("clone this", "Role: Who clones?\nContext: What repo?\nFocus: Directory or depth?\nOutput: Clone path or confirmation?"),
    ("init this", "Role: Who initializes?\nContext: What repo?\nFocus: Default branch or config?\nOutput: Init log or confirmation?"),
    ("config this", "Role: Who configures?\nContext: What tool?\nFocus: Key or value?\nOutput: Config file or confirmation?"),
    ("set this", "Role: Who sets?\nContext: What variable?\nFocus: Value or scope?\nOutput: Confirmation or value?"),
    ("get this", "Role: Who gets?\nContext: What value?\nFocus: Key or default?\nOutput: Value or error?"),
    ("list this", "Role: Who lists?\nContext: What items?\nFocus: Filter or sort?\nOutput: List or table?"),
    ("show this", "Role: Who shows?\nContext: What item?\nFocus: Details or view?\nOutput: Item or page?"),
    ("hide this", "Role: Who hides?\nContext: What item?\nFocus: Reason or duration?\nOutput: Confirmation or status?"),
    ("lock this", "Role: Who locks?\nContext: What resource?\nFocus: Reason or duration?\nOutput: Lock status or confirmation?"),
    ("unlock this", "Role: Who unlocks?\nContext: What resource?\nFocus: Reason or key?\nOutput: Unlock status or confirmation?"),
    ("archive this", "Role: Who archives?\nContext: What item?\nFocus: Reason or retention?\nOutput: Archive location or confirmation?"),
    ("unarchive this", "Role: Who unarchives?\nContext: What item?\nFocus: Reason or location?\nOutput: Restored item or confirmation?"),
    ("backup this", "Role: Who backs up?\nContext: What data?\nFocus: Schedule or destination?\nOutput: Backup log or confirmation?"),
    ("restore this", "Role: Who restores?\nContext: What backup?\nFocus: Point in time or target?\nOutput: Restore log or confirmation?"),
    ("snapshot this", "Role: Who snapshots?\nContext: What volume?\nFocus: Name or retention?\nOutput: Snapshot ID or confirmation?"),
    ("replicate this", "Role: Who replicates?\nContext: What data?\nFocus: Target or frequency?\nOutput: Replication log or confirmation?"),
    ("sync this", "Role: Who syncs?\nContext: What files?\nFocus: Direction or conflicts?\nOutput: Sync log or confirmation?"),
    ("mirror this", "Role: Who mirrors?\nContext: What site?\nFocus: Location or schedule?\nOutput: Mirror URL or confirmation?"),
    ("cache this", "Role: Who caches?\nContext: What data?\nFocus: TTL or strategy?\nOutput: Cache status or confirmation?"),
    ("purge this", "Role: Who purges?\nContext: What cache?\nFocus: Pattern or all?\nOutput: Purge log or confirmation?"),
    ("flush this", "Role: Who flushes?\nContext: What buffer?\nFocus: Reason or target?\nOutput: Flush log or confirmation?"),
    ("clear this", "Role: Who clears?\nContext: What data?\nFocus: Scope or confirmation?\nOutput: Clear log or confirmation?"),
    ("reset this", "Role: Who resets?\nContext: What item?\nFocus: Defaults or confirmation?\nOutput: Reset log or confirmation?"),
    ("restart this", "Role: Who restarts?\nContext: What service?\nFocus: Reason or window?\nOutput: Restart log or confirmation?"),
    ("reload this", "Role: Who reloads?\nContext: What config?\nFocus: Graceful or hard?\nOutput: Reload log or confirmation?"),
    ("refresh this", "Role: Who refreshes?\nContext: What data?\nFocus: Source or frequency?\nOutput: Refresh log or confirmation?"),
    ("update this", "Role: Who updates?\nContext: What record?\nFocus: Fields or validation?\nOutput: Update log or confirmation?"),
    ("create this", "Role: Who creates?\nContext: What item?\nFocus: Attributes or template?\nOutput: Created item or ID?"),
    ("read this", "Role: Who reads?\nContext: What item?\nFocus: Fields or format?\nOutput: Item or content?"),
    ("write this", "Role: Who writes?\nContext: What item?\nFocus: Content or format?\nOutput: Written item or ID?"),
    ("delete this", "Role: Who deletes?\nContext: What item?\nFocus: Confirmation or backup?\nOutput: Delete log or confirmation?"),
]

# ---------------------------------------------------------------------------
# Part 2: Clear prompts -> R.C.F.O formatted prompts
# Each pair teaches Inspector to decompose an already-clear request into the
# four R.C.F.O components so it can be acted upon precisely.
# ---------------------------------------------------------------------------
CLEAR_TO_RCFO: List[Tuple[str, str]] = [
    (
        "As a senior Python developer, refactor this legacy authentication module to use OAuth2 and return a JSON report with security findings.",
        "Role: Senior Python Developer\nContext: A legacy authentication module needs to be modernized to support OAuth2\nFocus: Refactor the authentication module to implement OAuth2\nOutput: JSON report with security findings",
    ),
    (
        "As a technical writer for a SaaS company, create a user onboarding guide for new customers in the context of our dashboard product, focusing on step-by-step setup instructions, and output a Markdown file.",
        "Role: Technical Writer for a SaaS company\nContext: New customers need to get started with the dashboard product\nFocus: Create a user onboarding guide with step-by-step setup instructions\nOutput: Markdown file",
    ),
    (
        "As a data analyst, analyze Q3 sales figures from the regional CSV file and identify underperforming regions, then produce a one-page executive summary in PDF.",
        "Role: Data Analyst\nContext: Q3 sales figures are available in a regional CSV file\nFocus: Analyze sales figures and identify underperforming regions\nOutput: One-page executive summary in PDF",
    ),
    (
        "As a UX designer, audit the checkout flow of our e-commerce mobile app, focusing on cart abandonment points, and deliver a wireframe with recommended fixes in Figma.",
        "Role: UX Designer\nContext: E-commerce mobile app checkout flow\nFocus: Audit the checkout flow and identify cart abandonment points\nOutput: Wireframe with recommended fixes in Figma",
    ),
    (
        "As a DevOps engineer, migrate our production Kubernetes cluster to a new region with zero downtime and document the runbook in Confluence.",
        "Role: DevOps Engineer\nContext: Production Kubernetes cluster migration to a new region\nFocus: Migrate the cluster with zero downtime\nOutput: Documented runbook in Confluence",
    ),
    (
        "As a customer support lead, draft a response template for refund requests that includes empathy, policy explanation, and next steps, suitable for email and chat.",
        "Role: Customer Support Lead\nContext: Refund requests from customers\nFocus: Draft a response template with empathy, policy explanation, and next steps\nOutput: Template suitable for email and chat",
    ),
    (
        "As a cybersecurity analyst, review the latest firewall logs for suspicious external IPs and generate a weekly threat report in Excel.",
        "Role: Cybersecurity Analyst\nContext: Latest firewall logs available\nFocus: Review logs for suspicious external IPs\nOutput: Weekly threat report in Excel",
    ),
    (
        "As a product manager, write a product requirements document for a dark-mode feature, including user stories, acceptance criteria, and success metrics, in Notion.",
        "Role: Product Manager\nContext: New dark-mode feature to be built\nFocus: Write a product requirements document with user stories, acceptance criteria, and success metrics\nOutput: Document in Notion",
    ),
    (
        "As a marketing manager, launch a Q4 email campaign for our summer collection, targeting segmented customers, and measure results in Mailchimp.",
        "Role: Marketing Manager\nContext: Q4 email campaign for summer collection\nFocus: Launch campaign to segmented customers\nOutput: Measured results in Mailchimp",
    ),
    (
        "As a finance analyst, build a 12-month cash flow forecast for the startup using the provided spreadsheet model, and present variances to the board.",
        "Role: Finance Analyst\nContext: 12-month cash flow forecast for a startup\nFocus: Build forecast and present variances\nOutput: Board presentation",
    ),
    (
        "As a sales director, create a quarterly sales playbook for the enterprise team, including objection handling and negotiation scripts, stored in SharePoint.",
        "Role: Sales Director\nContext: Quarterly sales playbook for enterprise team\nFocus: Objection handling and negotiation scripts\nOutput: Stored in SharePoint",
    ),
    (
        "As an HR business partner, design a 30-60-90 day onboarding plan for new software engineers, including milestones and check-ins, hosted in Notion.",
        "Role: HR Business Partner\nContext: New software engineer onboarding\nFocus: 30-60-90 day plan with milestones\nOutput: Hosted in Notion",
    ),
    (
        "As a cloud architect, design a multi-region AWS architecture for a fintech application, emphasizing security and compliance, and document it in a whitepaper.",
        "Role: Cloud Architect\nContext: Multi-region AWS architecture for fintech app\nFocus: Security and compliance emphasis\nOutput: Whitepaper",
    ),
    (
        "As a content strategist, develop a 6-month editorial calendar for the company blog, aligning with product launches and SEO targets, managed in Airtable.",
        "Role: Content Strategist\nContext: 6-month editorial calendar for company blog\nFocus: Align with product launches and SEO\nOutput: Managed in Airtable",
    ),
    (
        "As a quality assurance engineer, write automated test cases for the payment checkout flow using Selenium, targeting Chrome and Firefox, and store results in TestRail.",
        "Role: QA Engineer\nContext: Payment checkout flow\nFocus: Automated Selenium tests for Chrome/Firefox\nOutput: Results stored in TestRail",
    ),
    (
        "As a business intelligence analyst, build a Tableau dashboard visualizing customer churn by cohort and region, and schedule weekly email delivery to stakeholders.",
        "Role: BI Analyst\nContext: Customer churn by cohort and region\nFocus: Tableau dashboard\nOutput: Weekly email delivery to stakeholders",
    ),
    (
        "As a legal counsel, review the vendor service agreement for liability clauses and data protection terms, and provide a redlined version with comments.",
        "Role: Legal Counsel\nContext: Vendor service agreement\nFocus: Liability and data protection clauses\nOutput: Redlined version with comments",
    ),
    (
        "As a medical writer, draft a clinical trial protocol for a Phase II oncology study, including endpoints, inclusion criteria, and statistical methods, in Word.",
        "Role: Medical Writer\nContext: Phase II oncology clinical trial\nFocus: Protocol with endpoints, criteria, and methods\nOutput: Word document",
    ),
    (
        "As a supply chain manager, optimize warehouse inventory levels for Q4 using historical demand data, and present recommendations to operations leadership.",
        "Role: Supply Chain Manager\nContext: Q4 warehouse inventory optimization\nFocus: Historical demand data\nOutput: Recommendations to operations leadership",
    ),
    (
        "As a brand manager, refresh the company visual identity guidelines, including logo usage, color palette, and typography, distributed as a brand kit PDF.",
        "Role: Brand Manager\nContext: Company visual identity refresh\nFocus: Logo, color, typography\nOutput: Brand kit PDF",
    ),
    (
        "As a research scientist, reproduce the results from the published machine learning paper using the provided code repository, and document discrepancies in a Jupyter notebook.",
        "Role: Research Scientist\nContext: ML paper reproduction\nFocus: Code repository results\nOutput: Jupyter notebook with discrepancies",
    ),
    (
        "As a technical recruiter, source 20 senior backend engineer candidates for the open role, screen them, and present a shortlist with evaluation notes to the hiring manager.",
        "Role: Technical Recruiter\nContext: Senior backend engineer role\nFocus: Source, screen, shortlist\nOutput: Shortlist with evaluation notes",
    ),
    (
        "As an event planner, organize the annual company summit for 500 attendees, including venue, catering, and agenda, tracked in a master project plan.",
        "Role: Event Planner\nContext: Annual company summit for 500 attendees\nFocus: Venue, catering, agenda\nOutput: Master project plan",
    ),
    (
        "As a social media manager, respond to customer comments and direct messages on Instagram and Facebook following the community guidelines, using Sprout Social.",
        "Role: Social Media Manager\nContext: Instagram and Facebook comments/DMs\nFocus: Community guidelines compliance\nOutput: Responses via Sprout Social",
    ),
    (
        "As an e-commerce manager, audit product pages for SEO best practices, including meta titles, descriptions, and image alt text, and implement changes in Shopify.",
        "Role: E-commerce Manager\nContext: Product page SEO audit\nFocus: Meta titles, descriptions, alt text\nOutput: Changes implemented in Shopify",
    ),
    (
        "As a mobile developer, integrate push notifications into the iOS app using Firebase Cloud Messaging, handle token registration, and test on physical devices.",
        "Role: Mobile Developer\nContext: iOS app push notifications\nFocus: Firebase integration and testing\nOutput: Tested on physical devices",
    ),
    (
        "As a data engineer, build an ETL pipeline that extracts data from Salesforce, transforms it in dbt, and loads it into Snowflake, scheduled daily via Airflow.",
        "Role: Data Engineer\nContext: Salesforce to Snowflake ETL\nFocus: Extract, transform, load\nOutput: Scheduled daily via Airflow",
    ),
    (
        "As a copywriter, write ad copy for three Google Search ad groups promoting the new software trial, including headlines, descriptions, and display URLs.",
        "Role: Copywriter\nContext: Google Search ads for software trial\nFocus: Headlines, descriptions, URLs\nOutput: Three ad groups copy",
    ),
    (
        "As a systems administrator, apply security patches to all Linux servers in the staging environment, verify stability, and document changes in the changelog.",
        "Role: Systems Administrator\nContext: Linux servers in staging\nFocus: Security patches\nOutput: Documented in changelog",
    ),
    (
        "As a research assistant, conduct a literature review on remote work productivity published in the last 3 years, summarize key findings, and cite sources in APA format.",
        "Role: Research Assistant\nContext: Remote work productivity literature\nFocus: Last 3 years, key findings\nOutput: APA-cited summary",
    ),
    (
        "As a graphic designer, create a set of 10 social media graphics for the product launch campaign, following brand guidelines, and export them in PNG and SVG.",
        "Role: Graphic Designer\nContext: Product launch campaign\nFocus: 10 social media graphics\nOutput: PNG and SVG files",
    ),
    (
        "As a customer success manager, conduct quarterly business reviews for top 10 enterprise accounts, prepare health score reports, and identify expansion opportunities.",
        "Role: CSM\nContext: QBRs for top 10 enterprise accounts\nFocus: Health scores, expansion\nOutput: Reports and opportunities",
    ),
    (
        "As a machine learning engineer, fine-tune a BERT model for sentiment analysis on customer support tickets, evaluate on hold-out data, and log metrics in MLflow.",
        "Role: ML Engineer\nContext: BERT fine-tuning for sentiment\nFocus: Support tickets, hold-out evaluation\nOutput: Metrics logged in MLflow",
    ),
    (
        "As a procurement specialist, negotiate pricing with three cloud providers for a 12-month committed use discount, compare total cost of ownership, and recommend a vendor.",
        "Role: Procurement Specialist\nContext: Cloud provider pricing negotiation\nFocus: TCO comparison, vendor recommendation\nOutput: Recommendation document",
    ),
    (
        "As a technical writer, convert the legacy user manual from PDF to an online help center using Zendesk, reorganize by user task, and add searchable keywords.",
        "Role: Technical Writer\nContext: Legacy manual to Zendesk help center\nFocus: Reorganize by task, keywords\nOutput: Online help center",
    ),
    (
        "As a product designer, prototype a new mobile onboarding flow in Figma, conduct usability testing with 5 users, and iterate based on feedback.",
        "Role: Product Designer\nContext: Mobile onboarding flow\nFocus: Figma prototype, 5 user tests\nOutput: Iterated prototype",
    ),
    (
        "As a compliance officer, ensure all marketing materials comply with GDPR and CCPA requirements, review consent mechanisms, and sign off on campaign launch.",
        "Role: Compliance Officer\nContext: GDPR/CCPA marketing compliance\nFocus: Consent mechanisms\nOutput: Campaign launch sign-off",
    ),
    (
        "As a backend developer, implement rate limiting for the public API using Redis, configure thresholds per endpoint, and monitor usage via Grafana dashboards.",
        "Role: Backend Developer\nContext: Public API rate limiting\nFocus: Redis, thresholds, monitoring\nOutput: Grafana dashboards",
    ),
    (
        "As a data scientist, develop a churn prediction model using historical subscription data, perform feature selection, and report feature importance to stakeholders.",
        "Role: Data Scientist\nContext: Churn prediction model\nFocus: Feature selection, importance\nOutput: Stakeholder report",
    ),
    (
        "As a front-end developer, migrate the legacy jQuery codebase to React, preserving existing functionality, and deploy the updated frontend to Vercel.",
        "Role: Front-end Developer\nContext: jQuery to React migration\nFocus: Preserve functionality\nOutput: Deployed to Vercel",
    ),
    (
        "As an office manager, coordinate the office relocation to the new headquarters, manage vendor contracts, and communicate the move timeline to all employees.",
        "Role: Office Manager\nContext: Office relocation\nFocus: Vendors, timeline communication\nOutput: Move coordination",
    ),
    (
        "As a video editor, edit the 30-minute product demo video, add subtitles and background music, and export in 1080p MP4 for YouTube upload.",
        "Role: Video Editor\nContext: 30-minute product demo\nFocus: Subtitles, music, export\nOutput: 1080p MP4 for YouTube",
    ),
    (
        "As a podcast producer, book guests for the next season of the leadership podcast, coordinate recording schedules, and edit episodes for release.",
        "Role: Podcast Producer\nContext: Leadership podcast season\nFocus: Guest booking, scheduling, editing\nOutput: Edited episodes",
    ),
    (
        "As a community manager, moderate the Discord server for the open-source project, enforce community guidelines, and welcome new members daily.",
        "Role: Community Manager\nContext: Open-source Discord server\nFocus: Moderation, welcoming\nOutput: Daily community management",
    ),
    (
        "As a translator, localize the product website from English to Brazilian Portuguese, including all UI strings and legal pages, using a translation management system.",
        "Role: Translator\nContext: Website localization EN -> PT-BR\nFocus: UI strings, legal pages\nOutput: Localized via TMS",
    ),
    (
        "As a UX researcher, conduct contextual interviews with 12 power users to understand workflow pain points, synthesize findings into an affinity map, and present to product.",
        "Role: UX Researcher\nContext: Contextual interviews with 12 users\nFocus: Workflow pain points\nOutput: Affinity map, presentation",
    ),
    (
        "As an IT support specialist, resolve 20 open tickets in the help desk queue, prioritize by SLA, and document solutions in the internal knowledge base.",
        "Role: IT Support Specialist\nContext: 20 open help desk tickets\nFocus: SLA prioritization\nOutput: Knowledge base documentation",
    ),
    (
        "As a financial planner, create a retirement savings projection for a 35-year-old client, model multiple scenarios, and deliver a personalized financial plan PDF.",
        "Role: Financial Planner\nContext: Retirement projection for client\nFocus: Multiple scenarios\nOutput: Personalized financial plan PDF",
    ),
    (
        "As a real estate agent, prepare a comparative market analysis for the listed property, including comparable sales and pricing recommendation, for the seller.",
        "Role: Real Estate Agent\nContext: CMA for listed property\nFocus: Comparable sales, pricing\nOutput: CMA for seller",
    ),
    (
        "As a graphic designer, design a 24-page annual report for the non-profit organization, including impact metrics, financials, and donor recognition, print-ready.",
        "Role: Graphic Designer\nContext: 24-page annual report\nFocus: Impact, financials, donors\nOutput: Print-ready PDF",
    ),
    (
        "As a systems analyst, document the current state of the order management system, map data flows, and identify integration points with the new CRM.",
        "Role: Systems Analyst\nContext: Order management system\nFocus: Data flows, CRM integration\nOutput: Documentation",
    ),
    (
        "As a social media analyst, benchmark competitor Instagram engagement rates over the last 6 months, identify top-performing content types, and recommend strategy adjustments.",
        "Role: Social Media Analyst\nContext: Competitor Instagram benchmark\nFocus: Engagement rates, content types\nOutput: Strategy recommendations",
    ),
    (
        "As a technical program manager, lead the cross-functional launch of the mobile banking feature, coordinate engineering, design, and compliance teams, and track risks in Jira.",
        "Role: TPM\nContext: Mobile banking feature launch\nFocus: Cross-functional coordination\nOutput: Risk tracking in Jira",
    ),
    (
        "As a senior backend engineer, design a rate-limiting middleware for the Go API using Redis, configure per-endpoint thresholds, and expose metrics via Prometheus.",
        "Role: Senior Backend Engineer\nContext: Go API rate limiting\nFocus: Redis middleware, Prometheus metrics\nOutput: Middleware code and metrics dashboard",
    ),
    (
        "As a frontend tech lead, migrate the AngularJS dashboard to React with TypeScript, preserve all existing charts, and deploy to AWS S3 with CloudFront.",
        "Role: Frontend Tech Lead\nContext: AngularJS to React migration\nFocus: Preserve charts, deploy to S3/CloudFront\nOutput: Deployed React dashboard",
    ),
    (
        "As a data analyst, build a monthly recurring revenue (MRR) report from Stripe exports, segment by plan type, and deliver a Google Slides deck to the CFO.",
        "Role: Data Analyst\nContext: Stripe MRR data\nFocus: MRR by plan type\nOutput: Google Slides deck for CFO",
    ),
    (
        "As a product owner, prioritize the Q2 backlog using RICE scoring, write user stories for the top 10 items, and present the sprint plan in Jira.",
        "Role: Product Owner\nContext: Q2 backlog prioritization\nFocus: RICE scoring, top 10 user stories\nOutput: Sprint plan in Jira",
    ),
    (
        "As a security engineer, conduct a penetration test on the staging environment, document critical vulnerabilities, and provide remediation steps in a Confluence page.",
        "Role: Security Engineer\nContext: Staging environment pentest\nFocus: Critical vulnerabilities\nOutput: Confluence remediation page",
    ),
    (
        "As a UX writer, rewrite the error messages in the checkout flow to be clearer and more actionable, following the voice and tone guide, and submit a Figma file for review.",
        "Role: UX Writer\nContext: Checkout flow error messages\nFocus: Clarity and actionability\nOutput: Figma file for review",
    ),
    (
        "As a database administrator, tune slow queries in the PostgreSQL production database, add missing indexes, and report performance improvements to the engineering team.",
        "Role: Database Administrator\nContext: PostgreSQL production\nFocus: Slow query tuning\nOutput: Performance improvement report",
    ),
    (
        "As a business analyst, gather requirements for the new client portal from stakeholders, document functional and non-functional requirements, and obtain sign-off in a shared doc.",
        "Role: Business Analyst\nContext: New client portal requirements\nFocus: Functional and non-functional docs\nOutput: Signed-off requirements doc",
    ),
    (
        "As a content manager, publish 4 blog posts per week on the company website, optimize each for SEO, and track performance in Google Analytics.",
        "Role: Content Manager\nContext: Company blog\nFocus: 4 posts/week, SEO\nOutput: GA performance tracking",
    ),
    (
        "As an account manager, prepare the quarterly business review for the top enterprise client, including usage metrics and renewal discussion points, and deliver a PDF deck.",
        "Role: Account Manager\nContext: QBR for top enterprise client\nFocus: Usage metrics, renewal points\nOutput: PDF deck",
    ),
    (
        "As a recruiter, screen 50 applications for the senior designer role, conduct 10 first-round interviews, and recommend 3 finalists to the hiring manager.",
        "Role: Recruiter\nContext: Senior designer hiring\nFocus: Screen 50, interview 10, recommend 3\nOutput: Finalist recommendations",
    ),
    (
        "As a training coordinator, schedule the quarterly compliance training for all 200 employees, track completion in the LMS, and send reminders to laggards.",
        "Role: Training Coordinator\nContext: Quarterly compliance training\nFocus: 200 employees, LMS tracking\nOutput: Completion report",
    ),
    (
        "As a facilities manager, oversee the office move to the new building, coordinate with movers and IT, and ensure all workstations are ready by move-in day.",
        "Role: Facilities Manager\nContext: Office move to new building\nFocus: Movers, IT, workstations\nOutput: Move-in readiness report",
    ),
    (
        "As a procurement officer, source office supplies from three vendors, compare prices and delivery times, and place the order with the best value option.",
        "Role: Procurement Officer\nContext: Office supplies sourcing\nFocus: Price and delivery comparison\nOutput: Purchase order",
    ),
    (
        "As an event coordinator, plan the monthly team happy hour, select the venue, send invitations, and collect RSVPs by the deadline.",
        "Role: Event Coordinator\nContext: Monthly team happy hour\nFocus: Venue, invitations, RSVPs\nOutput: Event plan and RSVP list",
    ),
    (
        "As a social media coordinator, create and schedule 12 tweets for the product launch week, monitor mentions, and engage with replies daily.",
        "Role: Social Media Coordinator\nContext: Product launch week tweets\nFocus: 12 tweets, monitoring, engagement\nOutput: Scheduled tweets and engagement log",
    ),
    (
        "As a graphic design intern, resize the campaign banner into 5 different formats for web and social, follow the brand kit, and deliver the files in a shared folder.",
        "Role: Graphic Design Intern\nContext: Campaign banner resizing\nFocus: 5 formats, brand kit\nOutput: Resized banner files",
    ),
    (
        "As a copywriting intern, draft 3 variations of the homepage hero copy, run an A/B test, and report the winning variation to the marketing lead.",
        "Role: Copywriting Intern\nContext: Homepage hero copy\nFocus: 3 variations, A/B test\nOutput: A/B test results",
    ),
    (
        "As a video production assistant, log and tag the raw footage from the client shoot, organize it by scene in the editing software, and hand it off to the lead editor.",
        "Role: Video Production Assistant\nContext: Client shoot footage\nFocus: Log, tag, organize by scene\nOutput: Organized footage handoff",
    ),
    (
        "As a customer service representative, answer 30+ inbound calls daily, resolve billing inquiries using the knowledge base, and escalate complex cases to the team lead.",
        "Role: Customer Service Representative\nContext: Inbound support calls\nFocus: Billing inquiries, escalation\nOutput: Resolved cases and escalations",
    ),
    (
        "As a sales development representative, make 50 cold calls per day, qualify leads using the BANT framework, and book 5 qualified demos per week for account executives.",
        "Role: SDR\nContext: Cold calling\nFocus: BANT qualification, 5 demos/week\nOutput: Qualified demo bookings",
    ),
    (
        "As an account executive, run product demos for qualified prospects, negotiate contract terms, and close 2 new enterprise deals per quarter.",
        "Role: Account Executive\nContext: Product demos and negotiations\nFocus: Close 2 enterprise deals/quarter\nOutput: Signed contracts",
    ),
    (
        "As a customer success specialist, conduct quarterly check-ins with 25 mid-market accounts, monitor health scores, and flag at-risk accounts to the CS manager.",
        "Role: Customer Success Specialist\nContext: 25 mid-market accounts\nFocus: Health scores, at-risk flags\nOutput: At-risk account report",
    ),
    (
        "As a technical support engineer, troubleshoot customer VPN connectivity issues, document resolution steps in the ticketing system, and maintain a 90% first-contact resolution rate.",
        "Role: Technical Support Engineer\nContext: VPN connectivity issues\nFocus: Resolution steps, FCR rate\nOutput: Documented tickets",
    ),
    (
        "As a QA analyst, execute regression test cases for the bi-weekly release, log defects in Jira, and verify fixes before production deployment.",
        "Role: QA Analyst\nContext: Bi-weekly release regression\nFocus: Defect logging, fix verification\nOutput: Verified release candidates",
    ),
    (
        "As a release manager, coordinate the monthly production release, obtain sign-offs from engineering and QA, and communicate the release notes to all stakeholders.",
        "Role: Release Manager\nContext: Monthly production release\nFocus: Sign-offs, release notes\nOutput: Communicated release notes",
    ),
    (
        "As a site reliability engineer, monitor system uptime and latency SLOs, respond to PagerDuty incidents, and conduct post-mortems for Sev-1 events.",
        "Role: SRE\nContext: System uptime/latency SLOs\nFocus: Incident response, post-mortems\nOutput: Post-mortem reports",
    ),
    (
        "As a network engineer, configure VLANs and firewall rules for the new branch office, test connectivity, and document the network topology in Visio.",
        "Role: Network Engineer\nContext: New branch office network\nFocus: VLANs, firewall rules\nOutput: Visio network topology",
    ),
    (
        "As a system administrator, manage user accounts and permissions in Active Directory, onboard new hires, and offboard departing employees within 24 hours.",
        "Role: System Administrator\nContext: Active Directory management\nFocus: Onboarding/offboarding SLA\nOutput: AD change log",
    ),
    (
        "As a help desk technician, resolve Level 1 hardware and software issues for 150 employees, maintain asset inventory in ServiceNow, and achieve 4-hour average resolution.",
        "Role: Help Desk Technician\nContext: Level 1 support for 150 employees\nFocus: Asset inventory, resolution SLA\nOutput: ServiceNow ticket log",
    ),
    (
        "As a desktop support specialist, image new laptops with the corporate standard image, install required software, and deliver devices to new hires on their first day.",
        "Role: Desktop Support Specialist\nContext: Laptop imaging and delivery\nFocus: Corporate image, first-day delivery\nOutput: Imaged and delivered devices",
    ),
    (
        "As a procurement analyst, analyze spend data by category and vendor, identify cost-saving opportunities, and present recommendations to the CFO quarterly.",
        "Role: Procurement Analyst\nContext: Spend data analysis\nFocus: Cost-saving opportunities\nOutput: Quarterly recommendations",
    ),
    (
        "As an accounts payable clerk, process 100+ vendor invoices weekly, match them to purchase orders, and schedule payments within net-30 terms.",
        "Role: Accounts Payable Clerk\nContext: Vendor invoice processing\nFocus: PO matching, net-30 payments\nOutput: Processed invoices and payments",
    ),
    (
        "As an accounts receivable specialist, send monthly statements to 200+ customers, follow up on overdue invoices, and apply payments to the correct accounts.",
        "Role: Accounts Receivable Specialist\nContext: Customer statements and collections\nFocus: Overdue follow-up, payment application\nOutput: Collections report",
    ),
    (
        "As a payroll administrator, process bi-weekly payroll for 300 employees, verify timecard approvals, and file payroll taxes with federal and state agencies.",
        "Role: Payroll Administrator\nContext: Bi-weekly payroll for 300 employees\nFocus: Timecard verification, tax filing\nOutput: Processed payroll and filings",
    ),
    (
        "As a benefits administrator, manage open enrollment for health, dental, and vision plans, answer employee questions, and submit carrier enrollments by the deadline.",
        "Role: Benefits Administrator\nContext: Open enrollment management\nFocus: Carrier submissions, employee support\nOutput: Completed enrollments",
    ),
    (
        "As a compliance analyst, monitor regulatory changes affecting the company, assess impact on current policies, and update the compliance matrix quarterly.",
        "Role: Compliance Analyst\nContext: Regulatory monitoring\nFocus: Policy impact assessment\nOutput: Updated compliance matrix",
    ),
    (
        "As a risk analyst, identify operational risks in the loan origination process, assess likelihood and impact, and recommend mitigations to the risk committee.",
        "Role: Risk Analyst\nContext: Loan origination risks\nFocus: Likelihood and impact\nOutput: Risk mitigation recommendations",
    ),
    (
        "As an internal auditor, perform quarterly audits of expense reports, test compliance with company policy, and report findings to the audit committee.",
        "Role: Internal Auditor\nContext: Expense report audits\nFocus: Policy compliance testing\nOutput: Audit findings report",
    ),
    (
        "As a fraud analyst, review flagged transactions for potential fraud, contact customers to verify activity, and block compromised accounts within 1 hour.",
        "Role: Fraud Analyst\nContext: Flagged transaction review\nFocus: Customer verification, account blocking\nOutput: Fraud case dispositions",
    ),
    (
        "As an AML analyst, investigate suspicious activity reports, file SARs with FinCEN when required, and document investigation findings in the case management system.",
        "Role: AML Analyst\nContext: SAR investigation\nFocus: FinCEN filing, documentation\nOutput: SAR filings and case notes",
    ),
    (
        "As a KYC analyst, verify customer identity documents for new account openings, perform sanctions screening, and approve or reject applications within SLA.",
        "Role: KYC Analyst\nContext: New account KYC verification\nFocus: Document verification, sanctions\nOutput: Approved/rejected applications",
    ),
    (
        "As an underwriter, evaluate commercial loan applications, assess credit risk using financial statements, and approve or decline within the authority limit.",
        "Role: Underwriter\nContext: Commercial loan underwriting\nFocus: Credit risk assessment\nOutput: Approve/decline decisions",
    ),
    (
        "As a loan officer, meet with prospective borrowers, explain loan products and terms, and guide qualified applicants through the origination process.",
        "Role: Loan Officer\nContext: Prospective borrower meetings\nFocus: Product explanation, origination\nOutput: Loan applications in process",
    ),
    (
        "As a financial advisor, meet with clients to review investment portfolios, rebalance assets based on goals, and deliver quarterly performance reviews.",
        "Role: Financial Advisor\nContext: Client portfolio reviews\nFocus: Rebalancing, performance\nOutput: Quarterly performance reviews",
    ),
    (
        "As an insurance agent, quote auto and home insurance policies for new clients, explain coverage options, and bind policies upon client acceptance.",
        "Role: Insurance Agent\nContext: Auto and home insurance quoting\nFocus: Coverage options, binding\nOutput: Bound insurance policies",
    ),
    (
        "As a claims adjuster, investigate property damage claims, inspect damaged sites, and determine settlement amounts within policy limits.",
        "Role: Claims Adjuster\nContext: Property damage claims\nFocus: Site inspection, settlement\nOutput: Claim settlement determinations",
    ),
    (
        "As an underwriting assistant, collect supporting documents for loan applications, verify applicant information, and prepare application files for underwriter review.",
        "Role: Underwriting Assistant\nContext: Loan application documentation\nFocus: Verification, file preparation\nOutput: Complete application files",
    ),
    (
        "As a title examiner, search public records for property ownership history, identify liens or encumbrances, and issue title commitments for real estate transactions.",
        "Role: Title Examiner\nContext: Property title search\nFocus: Liens, encumbrances\nOutput: Title commitments",
    ),
    (
        "As a closing coordinator, schedule real estate closings, coordinate with buyers, sellers, and attorneys, and ensure all documents are signed and recorded.",
        "Role: Closing Coordinator\nContext: Real estate closings\nFocus: Document signing and recording\nOutput: Closed transactions",
    ),
    (
        "As a real estate assistant, list new properties on the MLS, schedule showings, and prepare marketing materials for agent review.",
        "Role: Real Estate Assistant\nContext: MLS listings and showings\nFocus: Marketing materials\nOutput: MLS listings and materials",
    ),
    (
        "As a leasing agent, show available rental units to prospective tenants, process rental applications, and execute lease agreements for approved applicants.",
        "Role: Leasing Agent\nContext: Rental unit showings\nFocus: Application processing, leases\nOutput: Executed lease agreements",
    ),
    (
        "As a property manager, collect monthly rent from 50 residential units, coordinate maintenance requests, and provide owners with monthly financial statements.",
        "Role: Property Manager\nContext: 50 residential units\nFocus: Rent collection, maintenance\nOutput: Monthly owner statements",
    ),
    (
        "As a maintenance technician, respond to maintenance requests within 24 hours, perform routine HVAC inspections, and log all work in the property management system.",
        "Role: Maintenance Technician\nContext: Maintenance requests and inspections\nFocus: 24-hour response, logging\nOutput: Completed work orders",
    ),
    (
        "As a groundskeeper, maintain landscaping for the commercial property, mow lawns weekly, and remove debris from common areas.",
        "Role: Groundskeeper\nContext: Commercial property landscaping\nFocus: Weekly mowing, debris removal\nOutput: Maintained property grounds",
    ),
    (
        "As a security guard, patrol the facility perimeter twice per shift, monitor CCTV feeds, and report suspicious activity to the dispatch center.",
        "Role: Security Guard\nContext: Facility perimeter patrol\nFocus: CCTV monitoring, reporting\nOutput: Incident reports",
    ),
    (
        "As a receptionist, answer and route incoming calls, greet visitors, and maintain the visitor log in accordance with security policy.",
        "Role: Receptionist\nContext: Front desk operations\nFocus: Call routing, visitor log\nOutput: Visitor log entries",
    ),
    (
        "As an administrative assistant, schedule executive meetings, prepare meeting minutes, and manage travel arrangements for the leadership team.",
        "Role: Administrative Assistant\nContext: Executive support\nFocus: Meeting minutes, travel\nOutput: Minutes and travel itineraries",
    ),
    (
        "As an executive assistant, manage the CEO's calendar, screen emails, and prepare briefing materials for board meetings.",
        "Role: Executive Assistant\nContext: CEO support\nFocus: Calendar, emails, briefings\nOutput: Briefing materials",
    ),
    (
        "As a personal assistant, run personal errands for the executive, manage household staff, and coordinate family schedules.",
        "Role: Personal Assistant\nContext: Executive personal support\nFocus: Errands, household, schedules\nOutput: Completed tasks and schedules",
    ),
    (
        "As a virtual assistant, manage email inboxes for 5 clients, schedule social media posts, and perform online research as requested.",
        "Role: Virtual Assistant\nContext: Remote admin support\nFocus: Email, social media, research\nOutput: Completed client tasks",
    ),
    (
        "As a freelance writer, write 2,000-word articles for client blogs, follow editorial guidelines, and submit drafts by the agreed deadline.",
        "Role: Freelance Writer\nContext: Client blog articles\nFocus: Word count, guidelines\nOutput: Submitted drafts",
    ),
    (
        "As a freelance editor, edit client manuscripts for grammar and clarity, track changes in Word, and return edited files within 5 business days.",
        "Role: Freelance Editor\nContext: Manuscript editing\nFocus: Grammar, clarity, track changes\nOutput: Edited manuscripts",
    ),
    (
        "As a freelance translator, translate legal documents from English to Spanish, maintain terminology consistency, and deliver certified translations.",
        "Role: Freelance Translator\nContext: Legal document translation\nFocus: Terminology, certification\nOutput: Certified translations",
    ),
    (
        "As a freelance designer, create logo concepts for startup clients, present 3 options per brief, and deliver final files in AI and PNG formats.",
        "Role: Freelance Designer\nContext: Logo design\nFocus: 3 concepts, file formats\nOutput: Final logo files",
    ),
    (
        "As a freelance developer, build custom WordPress themes for clients, ensure responsive design, and provide 30 days of post-launch support.",
        "Role: Freelance Developer\nContext: WordPress theme development\nFocus: Responsive design, support\nOutput: Custom themes and support",
    ),
    (
        "As a freelance photographer, photograph real estate properties for listing photos, edit images, and deliver the gallery within 48 hours.",
        "Role: Freelance Photographer\nContext: Real estate photography\nFocus: Image editing, 48-hour delivery\nOutput: Photo gallery",
    ),
    (
        "As a freelance videographer, shoot and edit promotional videos for local businesses, add music and captions, and deliver the final MP4 files.",
        "Role: Freelance Videographer\nContext: Promotional video production\nFocus: Music, captions, MP4 delivery\nOutput: Final videos",
    ),
    (
        "As a freelance consultant, assess the client's current marketing strategy, identify gaps, and present a 90-day improvement plan.",
        "Role: Freelance Consultant\nContext: Marketing strategy assessment\nFocus: Gap analysis, 90-day plan\nOutput: Improvement plan presentation",
    ),
    (
        "As a freelance coach, conduct weekly 1-on-1 coaching sessions with clients, track progress toward goals, and adjust action plans as needed.",
        "Role: Freelance Coach\nContext: Client coaching sessions\nFocus: Progress tracking, plan adjustment\nOutput: Client progress reports",
    ),
    (
        "As a freelance tutor, teach high school mathematics online, prepare lesson materials, and provide progress reports to parents monthly.",
        "Role: Freelance Tutor\nContext: Online math tutoring\nFocus: Lesson prep, progress reports\nOutput: Monthly progress reports",
    ),
    (
        "As a freelance transcriber, transcribe podcast episodes with speaker labels, deliver verbatim transcripts in Word, and meet the 24-hour turnaround.",
        "Role: Freelance Transcriber\nContext: Podcast transcription\nFocus: Speaker labels, verbatim\nOutput: Transcript files",
    ),
    (
        "As a freelance data entry specialist, enter survey responses into the database, verify accuracy against source forms, and maintain 99% accuracy rate.",
        "Role: Freelance Data Entry Specialist\nContext: Survey data entry\nFocus: Accuracy verification\nOutput: Verified database entries",
    ),
    (
        "As a freelance virtual bookkeeper, reconcile bank statements for small business clients, categorize transactions, and deliver monthly financial reports.",
        "Role: Freelance Virtual Bookkeeper\nContext: Small business bookkeeping\nFocus: Reconciliation, categorization\nOutput: Monthly financial reports",
    ),
    (
        "As a freelance tax preparer, prepare individual and small business tax returns, gather required documentation, and file electronically before the deadline.",
        "Role: Freelance Tax Preparer\nContext: Tax return preparation\nFocus: Documentation, e-filing\nOutput: Filed tax returns",
    ),
    (
        "As a freelance notary, witness signatures on legal documents, verify signer identity, and apply notary seal and signature.",
        "Role: Freelance Notary\nContext: Document notarization\nFocus: Identity verification\nOutput: Notarized documents",
    ),
    (
        "As a freelance event planner, plan corporate holiday parties for clients, select venues and caterers, and manage the event budget.",
        "Role: Freelance Event Planner\nContext: Corporate holiday parties\nFocus: Venue, caterer, budget\nOutput: Executed events",
    ),
    (
        "As a freelance personal shopper, source clothing and accessories for clients based on style preferences, arrange fittings, and process purchases.",
        "Role: Freelance Personal Shopper\nContext: Personal shopping\nFocus: Style preferences, fittings\nOutput: Purchased items",
    ),
    (
        "As a freelance meal planner, create weekly meal plans for clients based on dietary restrictions, generate grocery lists, and provide cooking instructions.",
        "Role: Freelance Meal Planner\nContext: Meal planning\nFocus: Dietary restrictions, grocery lists\nOutput: Meal plans and lists",
    ),
    (
        "As a freelance pet sitter, care for client pets while owners are away, provide daily updates and photos, and administer medications as needed.",
        "Role: Freelance Pet Sitter\nContext: Pet care\nFocus: Daily updates, medication\nOutput: Care logs and photos",
    ),
    (
        "As a freelance house cleaner, clean client homes according to the agreed checklist, use client-supplied products, and complete jobs within scheduled time.",
        "Role: Freelance House Cleaner\nContext: Home cleaning\nFocus: Checklist completion\nOutput: Cleaned homes",
    ),
    (
        "As a freelance gardener, maintain client gardens, mow lawns, prune shrubs, and remove weeds on a weekly schedule.",
        "Role: Freelance Gardener\nContext: Garden maintenance\nFocus: Weekly schedule\nOutput: Maintained gardens",
    ),
    (
        "As a freelance handyman, perform minor home repairs for clients, fix leaky faucets and patch walls, and bill by the hour.",
        "Role: Freelance Handyman\nContext: Minor home repairs\nFocus: Hourly billing\nOutput: Completed repairs",
    ),
    (
        "As a freelance mover, help clients pack and load belongings, transport items to the new location, and unload and place items as directed.",
        "Role: Freelance Mover\nContext: Moving services\nFocus: Packing, transport, unloading\nOutput: Completed moves",
    ),
    (
        "As a freelance driver, provide rideshare services through the platform, maintain vehicle cleanliness, and maintain a 4.8+ star rating.",
        "Role: Freelance Driver\nContext: Rideshare driving\nFocus: Cleanliness, rating\nOutput: Completed rides",
    ),
    (
        "As a freelance driver, provide rideshare services through the platform, maintain vehicle cleanliness, and maintain a 4.8+ star rating.",
        "Role: Freelance Driver\nContext: Rideshare driving\nFocus: Cleanliness, rating\nOutput: Completed rides",
    ),
    (
        "As a freelance delivery driver, deliver food orders to customers, follow delivery instructions, and maintain delivery temperature standards.",
        "Role: Freelance Delivery Driver\nContext: Food delivery\nFocus: Instructions, temperature\nOutput: Completed deliveries",
    ),
    # --- Additional Clear Prompts (expansion to 500+ total) --------------------
    (
        "As a senior full-stack developer, build a REST API for the inventory management system using Node.js and PostgreSQL, implement JWT authentication, and deploy to Heroku with a CI/CD pipeline.",
        "Role: Senior Full-stack Developer\nContext: Inventory management system\nFocus: REST API, JWT auth, CI/CD deploy\nOutput: Deployed API on Heroku",
    ),
    (
        "As a mobile QA lead, write test plans for the Android and iOS releases, execute manual regression tests, and log bugs in Azure DevOps for the sprint review.",
        "Role: Mobile QA Lead\nContext: Android and iOS releases\nFocus: Test plans, regression, bug logging\nOutput: Azure DevOps bug log",
    ),
    (
        "As a data steward, define data quality rules for the customer master data, monitor data quality dashboards, and escalate data issues to the data governance council.",
        "Role: Data Steward\nContext: Customer master data\nFocus: Data quality rules, dashboards\nOutput: Escalated data issues",
    ),
    (
        "As a cloud security engineer, configure AWS IAM roles and policies, enable CloudTrail logging, and conduct quarterly access reviews for all IAM users.",
        "Role: Cloud Security Engineer\nContext: AWS IAM configuration\nFocus: Roles, policies, access reviews\nOutput: Quarterly access review report",
    ),
    (
        "As a site reliability engineer, implement auto-scaling groups for the web tier, set up CloudWatch alarms, and reduce mean time to recovery by 30%.",
        "Role: Site Reliability Engineer\nContext: Web tier auto-scaling\nFocus: Auto-scaling groups, CloudWatch alarms\nOutput: Reduced MTTR by 30%",
    ),
    (
        "As a product marketing manager, create positioning and messaging for the new enterprise feature, train the sales team on value propositions, and launch the feature announcement campaign.",
        "Role: Product Marketing Manager\nContext: New enterprise feature\nFocus: Positioning, sales training, launch\nOutput: Feature announcement campaign",
    ),
    (
        "As a customer education manager, build a customer training academy in the LMS, create video tutorials for top 5 use cases, and achieve 80% course completion within 60 days.",
        "Role: Customer Education Manager\nContext: Customer training academy\nFocus: Video tutorials, completion rate\nOutput: LMS academy with 80% completion",
    ),
    (
        "As a technical account manager, manage relationships with 15 strategic enterprise accounts, conduct quarterly health checks, and identify upsell opportunities worth $2M ARR.",
        "Role: Technical Account Manager\nContext: 15 strategic enterprise accounts\nFocus: Health checks, upsell opportunities\nOutput: $2M ARR upsell pipeline",
    ),
    (
        "As a solutions architect, design a disaster recovery plan for the production environment, define RPO and RTO targets, and test failover procedures quarterly.",
        "Role: Solutions Architect\nContext: Disaster recovery plan\nFocus: RPO/RTO, failover testing\nOutput: Tested DR plan",
    ),
    (
        "As a data privacy officer, maintain the record of processing activities, respond to data subject access requests within 30 days, and report breaches to supervisory authorities within 72 hours.",
        "Role: Data Privacy Officer\nContext: Data privacy compliance\nFocus: DSAR responses, breach reporting\nOutput: Compliance records and reports",
    ),
    (
        "As a sustainability analyst, calculate the company's carbon footprint for Scope 1 and 2 emissions, identify reduction opportunities, and publish the annual sustainability report.",
        "Role: Sustainability Analyst\nContext: Carbon footprint calculation\nFocus: Scope 1 and 2 emissions\nOutput: Annual sustainability report",
    ),
    (
        "As a diversity and inclusion manager, analyze workforce diversity metrics, design inclusive hiring initiatives, and present progress to the executive diversity council quarterly.",
        "Role: Diversity and Inclusion Manager\nContext: Workforce diversity metrics\nFocus: Inclusive hiring initiatives\nOutput: Quarterly progress report to EDI council",
    ),
    (
        "As a learning experience designer, create interactive e-learning modules using Articulate Storyline, embed knowledge checks, and publish to the corporate LMS.",
        "Role: Learning Experience Designer\nContext: E-learning modules\nFocus: Articulate Storyline, knowledge checks\nOutput: Published LMS modules",
    ),
    (
        "As a talent acquisition specialist, build talent pipelines for hard-to-fill engineering roles, attend tech meetups and career fairs, and maintain a 30-day time-to-fill average.",
        "Role: Talent Acquisition Specialist\nContext: Engineering talent pipelines\nFocus: Sourcing, time-to-fill\nOutput: 30-day time-to-fill metric",
    ),
    (
        "As a compensation analyst, benchmark salary ranges against market data, recommend pay adjustments for high performers, and manage the annual merit increase cycle.",
        "Role: Compensation Analyst\nContext: Salary benchmarking\nFocus: Market data, merit cycle\nOutput: Compensation recommendations",
    ),
    (
        "As an employee relations manager, investigate employee relations cases, recommend disciplinary actions when warranted, and maintain compliance with labor laws.",
        "Role: Employee Relations Manager\nContext: Employee relations cases\nFocus: Investigations, labor law compliance\nOutput: Case resolutions and recommendations",
    ),
    (
        "As a workforce planner, forecast staffing needs for the next 12 months based on business growth projections, and present headcount plans to the finance and HR leadership.",
        "Role: Workforce Planner\nContext: 12-month staffing forecast\nFocus: Headcount plans\nOutput: Headcount plan presentation",
    ),
    (
        "As an internal communications manager, write the weekly CEO newsletter, announce company-wide policy changes, and maintain the intranet homepage content.",
        "Role: Internal Communications Manager\nContext: Company-wide communications\nFocus: CEO newsletter, intranet\nOutput: Weekly newsletter and intranet updates",
    ),
    (
        "As a change management lead, develop change management plans for the ERP implementation, conduct stakeholder impact assessments, and deliver training to affected teams.",
        "Role: Change Management Lead\nContext: ERP implementation\nFocus: Impact assessments, training\nOutput: Change management plans and training",
    ),
    (
        "As a business process analyst, document the current state procure-to-pay process, identify bottlenecks, and design the future state process with automation opportunities.",
        "Role: Business Process Analyst\nContext: Procure-to-pay process\nFocus: Bottlenecks, future state design\nOutput: Future state process design",
    ),
    (
        "As a systems integration specialist, integrate the CRM with the marketing automation platform using APIs, synchronize lead data, and validate data accuracy post-integration.",
        "Role: Systems Integration Specialist\nContext: CRM and marketing automation integration\nFocus: API integration, data sync\nOutput: Integrated systems with validated data",
    ),
    (
        "As a master data manager, govern the product master data, resolve duplicate records, and enforce data stewardship policies across the organization.",
        "Role: Master Data Manager\nContext: Product master data governance\nFocus: Duplicate resolution, stewardship\nOutput: Clean product master data",
    ),
    (
        "As a data warehouse architect, design the star schema for the sales data warehouse, implement slowly changing dimensions, and optimize query performance.",
        "Role: Data Warehouse Architect\nContext: Sales data warehouse\nFocus: Star schema, SCDs, query optimization\nOutput: Optimized data warehouse",
    ),
    (
        "As a data lake engineer, ingest raw data from multiple sources into the data lake using Azure Data Factory, organize data in the medallion architecture, and manage access controls.",
        "Role: Data Lake Engineer\nContext: Data lake ingestion\nFocus: Azure Data Factory, medallion architecture\nOutput: Organized data lake",
    ),
    (
        "As a machine learning operations engineer, deploy the recommendation model to production using MLflow and Kubernetes, monitor model drift, and retrain on a monthly schedule.",
        "Role: MLOps Engineer\nContext: Recommendation model deployment\nFocus: MLflow, Kubernetes, model drift\nOutput: Production ML pipeline",
    ),
    (
        "As a computer vision engineer, train an object detection model for quality inspection on the manufacturing line, label 5,000 images, and achieve 95% mAP on the test set.",
        "Role: Computer Vision Engineer\nContext: Manufacturing quality inspection\nFocus: Object detection, image labeling\nOutput: 95% mAP inspection model",
    ),
    (
        "As a natural language processing engineer, fine-tune a sentiment analysis model for customer reviews, evaluate precision and recall, and integrate the model into the feedback pipeline.",
        "Role: NLP Engineer\nContext: Customer review sentiment analysis\nFocus: Precision, recall, integration\nOutput: Integrated sentiment model",
    ),
    (
        "As a data visualization specialist, create interactive Power BI reports for the executive team, design drill-through pages, and schedule automated data refreshes.",
        "Role: Data Visualization Specialist\nContext: Power BI executive reports\nFocus: Drill-through pages, refreshes\nOutput: Interactive Power BI reports",
    ),
    (
        "As a business intelligence developer, write complex SQL queries for the marketing attribution model, build ETL views in dbt, and document the data lineage.",
        "Role: BI Developer\nContext: Marketing attribution model\nFocus: SQL queries, dbt views\nOutput: Documented data lineage",
    ),
    (
        "As a data analyst intern, clean and validate survey data in Excel, create pivot tables for summary statistics, and prepare charts for the research team presentation.",
        "Role: Data Analyst Intern\nContext: Survey data cleaning\nFocus: Pivot tables, charts\nOutput: Presentation-ready charts",
    ),
    (
        "As a product designer intern, create wireframes for the new settings page, conduct 3 user interviews to validate flow, and present findings to the design team.",
        "Role: Product Designer Intern\nContext: Settings page wireframes\nFocus: User interviews, validation\nOutput: Wireframe findings presentation",
    ),
    (
        "As a software engineering intern, fix 5 bugs in the ticketing system, write unit tests for the fixes, and submit pull requests for code review.",
        "Role: Software Engineering Intern\nContext: Ticketing system bugs\nFocus: Bug fixes, unit tests\nOutput: Submitted pull requests",
    ),
    (
        "As a marketing intern, schedule social media posts using Hootsuite, monitor campaign hashtags, and compile weekly engagement reports for the marketing manager.",
        "Role: Marketing Intern\nContext: Social media scheduling\nFocus: Hootsuite, engagement reports\nOutput: Weekly engagement reports",
    ),
    (
        "As a sales intern, update CRM records with call notes, research prospect companies on LinkedIn, and schedule discovery calls for account executives.",
        "Role: Sales Intern\nContext: CRM and prospect research\nFocus: Call notes, discovery calls\nOutput: Updated CRM and scheduled calls",
    ),
    (
        "As a human resources intern, screen entry-level resumes, schedule phone screens with candidates, and maintain the applicant tracking system records.",
        "Role: Human Resources Intern\nContext: Resume screening\nFocus: Phone screens, ATS records\nOutput: Screened candidates",
    ),
    (
        "As a finance intern, reconcile monthly expense reports, prepare expense summaries for department heads, and assist with the quarterly budget close.",
        "Role: Finance Intern\nContext: Expense reconciliation\nFocus: Summaries, budget close\nOutput: Expense summaries",
    ),
    (
        "As an operations intern, track daily warehouse inventory counts, update the inventory management system, and report discrepancies to the warehouse manager.",
        "Role: Operations Intern\nContext: Warehouse inventory tracking\nFocus: Count updates, discrepancy reporting\nOutput: Inventory discrepancy report",
    ),
    (
        "As a customer success intern, create customer health scorecards in the CRM, send welcome emails to new customers, and log support ticket interactions.",
        "Role: Customer Success Intern\nContext: Customer health scorecards\nFocus: Welcome emails, ticket logging\nOutput: CRM health scorecards",
    ),
    (
        "As a public relations intern, monitor media mentions of the company, compile daily press clippings, and draft press release distribution lists.",
        "Role: Public Relations Intern\nContext: Media monitoring\nFocus: Press clippings, distribution lists\nOutput: Daily press clippings",
    ),
    (
        "As a graphic design intern, create social media story templates in Canva, resize assets for different platforms, and organize files in the shared design folder.",
        "Role: Graphic Design Intern\nContext: Social media story templates\nFocus: Canva, asset resizing\nOutput: Organized design assets",
    ),
    (
        "As a video editing intern, trim raw interview footage, add lower-third graphics, and export 1-minute clips for social media promotion.",
        "Role: Video Editing Intern\nContext: Interview footage editing\nFocus: Lower-thirds, clip export\nOutput: Social media video clips",
    ),
    (
        "As a technical writing intern, update the API quickstart guide with new endpoint examples, verify all links work, and submit the revised guide for publication.",
        "Role: Technical Writing Intern\nContext: API quickstart guide\nFocus: Endpoint examples, link verification\nOutput: Revised quickstart guide",
    ),
    (
        "As a legal intern, summarize case law for the litigation team, organize exhibits in the case management system, and draft client intake memoranda.",
        "Role: Legal Intern\nContext: Case law summarization\nFocus: Exhibit organization, memoranda\nOutput: Case summaries and memos",
    ),
    (
        "As a medical coding intern, assign ICD-10 codes to patient charts, verify coding accuracy against coding guidelines, and submit coded charts for billing.",
        "Role: Medical Coding Intern\nContext: Medical chart coding\nFocus: ICD-10 coding, accuracy\nOutput: Coded patient charts",
    ),
    (
        "As a clinical research assistant, screen potential study participants, obtain informed consent, and enter study data into the electronic data capture system.",
        "Role: Clinical Research Assistant\nContext: Study participant screening\nFocus: Informed consent, data entry\nOutput: EDC study data entries",
    ),
    (
        "As a laboratory technician, process patient blood samples, run complete blood count tests, and report results to the ordering physician within 4 hours.",
        "Role: Laboratory Technician\nContext: Blood sample processing\nFocus: CBC tests, result reporting\nOutput: Lab results within 4 hours",
    ),
    (
        "As a radiology technologist, perform X-ray and CT scans per physician orders, ensure patient positioning and safety, and transmit images to PACS for radiologist review.",
        "Role: Radiology Technologist\nContext: X-ray and CT scans\nFocus: Patient safety, PACS transmission\nOutput: Transmitted imaging studies",
    ),
    (
        "As a pharmacy technician, receive and verify prescription orders, count and label medications, and process insurance claims through the pharmacy system.",
        "Role: Pharmacy Technician\nContext: Prescription processing\nFocus: Verification, labeling, claims\nOutput: Filled prescriptions",
    ),
    (
        "As a dental hygienist, perform dental cleanings and periodontal assessments, take dental X-rays, and educate patients on oral hygiene best practices.",
        "Role: Dental Hygienist\nContext: Dental cleanings\nFocus: X-rays, patient education\nOutput: Completed cleanings and education",
    ),
    (
        "As a physical therapist, develop individualized treatment plans for post-surgical patients, conduct therapy sessions, and document progress in the electronic health record.",
        "Role: Physical Therapist\nContext: Post-surgical rehabilitation\nFocus: Treatment plans, progress docs\nOutput: EHR progress notes",
    ),
    (
        "As an occupational therapist, assess patients' ability to perform daily living activities, recommend adaptive equipment, and document functional outcomes.",
        "Role: Occupational Therapist\nContext: Daily living assessment\nFocus: Adaptive equipment, outcomes\nOutput: Functional outcome documentation",
    ),
    (
        "As a speech-language pathologist, evaluate patients with swallowing disorders, develop swallowing therapy plans, and document progress toward safe oral intake.",
        "Role: Speech-Language Pathologist\nContext: Swallowing disorder evaluation\nFocus: Therapy plans, progress\nOutput: Swallowing therapy documentation",
    ),
    (
        "As a respiratory therapist, administer breathing treatments to ICU patients, manage ventilator settings, and respond to code blue emergencies.",
        "Role: Respiratory Therapist\nContext: ICU breathing treatments\nFocus: Ventilator management, emergencies\nOutput: Treatment and emergency response logs",
    ),
    (
        "As a dietitian, create personalized nutrition plans for patients with diabetes, conduct nutrition counseling sessions, and track patient weight and lab progress.",
        "Role: Dietitian\nContext: Diabetes nutrition plans\nFocus: Counseling, progress tracking\nOutput: Nutrition plans and progress logs",
    ),
    (
        "As a social worker, complete psychosocial assessments for hospital patients, connect patients with community resources, and facilitate discharge planning.",
        "Role: Social Worker\nContext: Psychosocial assessments\nFocus: Resource connection, discharge planning\nOutput: Completed discharge plans",
    ),
    (
        "As a case manager, coordinate care for high-risk patients, schedule follow-up appointments, and ensure medication adherence through regular check-ins.",
        "Role: Case Manager\nContext: High-risk patient care\nFocus: Appointments, adherence\nOutput: Coordinated care plans",
    ),
    (
        "As a school counselor, conduct individual counseling sessions with students, develop academic and career plans, and present wellness workshops to classrooms.",
        "Role: School Counselor\nContext: Student counseling\nFocus: Academic/career plans, workshops\nOutput: Student plans and workshop delivery",
    ),
    (
        "As a teacher, plan and deliver daily mathematics lessons for 8th grade, grade assignments within 48 hours, and maintain an up-to-date online gradebook.",
        "Role: Teacher\nContext: 8th grade mathematics\nFocus: Lesson delivery, grading\nOutput: Graded assignments and gradebook",
    ),
    (
        "As a professor, lecture on organic chemistry to undergraduate students, hold weekly office hours, and grade midterm and final examinations.",
        "Role: Professor\nContext: Organic chemistry lectures\nFocus: Office hours, exam grading\nOutput: Graded exams and student feedback",
    ),
    (
        "As a librarian, catalog new book acquisitions using the Dewey Decimal System, assist patrons with research queries, and manage the library lending system.",
        "Role: Librarian\nContext: Book cataloging\nFocus: Dewey Decimal, patron assistance\nOutput: Cataloged books and lending records",
    ),
    (
        "As an archivist, preserve historical documents in climate-controlled storage, create finding aids for collections, and digitize fragile materials for online access.",
        "Role: Archivist\nContext: Historical document preservation\nFocus: Finding aids, digitization\nOutput: Digital archive access",
    ),
    (
        "As a museum curator, plan the upcoming exhibition on modern art, select artworks from the collection, and write interpretive labels for visitors.",
        "Role: Museum Curator\nContext: Modern art exhibition\nFocus: Artwork selection, labels\nOutput: Exhibition and interpretive materials",
    ),
    (
        "As a journalist, investigate local government spending, interview key sources, and publish a 2,000-word investigative article in the newspaper.",
        "Role: Journalist\nContext: Government spending investigation\nFocus: Source interviews, article writing\nOutput: Published investigative article",
    ),
    (
        "As a copy editor, proofread manuscript chapters for grammar and style consistency, apply the house style guide, and return edited chapters to the author within one week.",
        "Role: Copy Editor\nContext: Manuscript proofreading\nFocus: Style guide consistency\nOutput: Edited chapters",
    ),
    (
        "As a translator, translate technical manuals from German to English, maintain industry terminology consistency, and deliver translations within agreed deadlines.",
        "Role: Translator\nContext: Technical manual translation\nFocus: Terminology, deadlines\nOutput: Delivered translations",
    ),
    (
        "As an interpreter, provide simultaneous interpretation for international conference sessions, use the interpretation booth equipment, and maintain confidentiality of session content.",
        "Role: Interpreter\nContext: Conference interpretation\nFocus: Simultaneous interpretation, confidentiality\nOutput: Interpreted conference sessions",
    ),
    (
        "As a voice actor, record narration for the corporate training video, deliver clean audio files in WAV format, and incorporate client feedback within 2 business days.",
        "Role: Voice Actor\nContext: Training video narration\nFocus: WAV audio, feedback incorporation\nOutput: Recorded narration files",
    ),
    (
        "As a musician, compose original background music for the mobile game, deliver stems in MP3 and WAV, and license the music under the agreed terms.",
        "Role: Musician\nContext: Game background music\nFocus: Stems, licensing\nOutput: Licensed music stems",
    ),
    (
        "As a sound designer, create sound effects for the animated short film, layer and mix audio tracks, and deliver the final sound mix in stereo WAV.",
        "Role: Sound Designer\nContext: Animated short film\nFocus: Sound effects, mixing\nOutput: Final stereo WAV mix",
    ),
    (
        "As an illustrator, create 10 illustrations for the children's book, follow the art direction, and deliver high-resolution files in TIFF format.",
        "Role: Illustrator\nContext: Children's book illustrations\nFocus: Art direction, TIFF delivery\nOutput: 10 high-res illustrations",
    ),
    (
        "As a 3D artist, model the product packaging design in Blender, apply textures and lighting, and render photorealistic images for the marketing website.",
        "Role: 3D Artist\nContext: Packaging design modeling\nFocus: Textures, lighting, rendering\nOutput: Photorealistic render images",
    ),
    (
        "As an animator, rig and animate the 3D character for the explainer video, sync animation to voiceover, and deliver the final MP4 animation.",
        "Role: Animator\nContext: Explainer video character animation\nFocus: Rigging, voiceover sync\nOutput: Final MP4 animation",
    ),
    (
        "As a motion graphics designer, create animated lower thirds and transitions for the news broadcast, follow the station's brand guidelines, and deliver After Effects project files.",
        "Role: Motion Graphics Designer\nContext: News broadcast graphics\nFocus: Brand guidelines, After Effects\nOutput: AE project files and graphics",
    ),
    (
        "As a colorist, color grade the commercial footage to match the brand look, apply consistent LUTs across all shots, and deliver the graded master in ProRes 422.",
        "Role: Colorist\nContext: Commercial footage grading\nFocus: LUTs, ProRes 422\nOutput: Graded master file",
    ),
    (
        "As a VFX artist, create visual effects for the science fiction short film, composite CGI elements with live action, and deliver the final shots in DPX format.",
        "Role: VFX Artist\nContext: Sci-fi short film VFX\nFocus: CGI compositing, DPX delivery\nOutput: Final VFX shots",
    ),
    (
        "As a game designer, design the core gameplay loop for the puzzle game, write the game design document, and balance difficulty across 50 levels.",
        "Role: Game Designer\nContext: Puzzle game design\nFocus: GDD, level balancing\nOutput: Game design document and balanced levels",
    ),
    (
        "As a game developer, implement the player controller in Unity, write C# scripts for game mechanics, and build the game for PC and mobile platforms.",
        "Role: Game Developer\nContext: Unity game development\nFocus: C# scripts, platform builds\nOutput: Built game for PC and mobile",
    ),
    (
        "As a game artist, create 2D sprite assets for the platformer game, animate character walk cycles, and deliver sprite sheets in PNG format.",
        "Role: Game Artist\nContext: Platformer sprite creation\nFocus: Walk cycles, sprite sheets\nOutput: PNG sprite sheets",
    ),
    (
        "As a level designer, build 10 playable levels for the adventure game, place enemies and collectibles, and test level flow with playtesters.",
        "Role: Level Designer\nContext: Adventure game levels\nFocus: Enemy placement, playtesting\nOutput: 10 tested levels",
    ),
    (
        "As a game producer, manage the game development schedule, track milestones in Jira, and coordinate between art, design, and engineering teams.",
        "Role: Game Producer\nContext: Game dev schedule management\nFocus: Milestone tracking, coordination\nOutput: Coordinated development milestones",
    ),
    (
        "As a game tester, playtest the beta version of the RPG, report 20+ bugs in the bug tracking system, and verify fixes in the next build.",
        "Role: Game Tester\nContext: RPG beta playtesting\nFocus: Bug reporting, fix verification\nOutput: 20+ reported bugs and verifications",
    ),
    (
        "As an esports coach, review match footage with the team, develop strategies for the upcoming tournament, and conduct practice sessions 4 times per week.",
        "Role: Esports Coach\nContext: Team strategy development\nFocus: Match review, practice sessions\nOutput: Tournament strategies and practice logs",
    ),
    (
        "As a streamer, stream gameplay for 4 hours daily on Twitch, engage with chat viewers, and maintain a consistent streaming schedule.",
        "Role: Streamer\nContext: Twitch streaming\nFocus: Viewer engagement, schedule\nOutput: Streamed hours and chat engagement",
    ),
    (
        "As a content creator, create short-form videos for TikTok and Instagram Reels, use trending audio and hashtags, and grow followers to 10,000 in 6 months.",
        "Role: Content Creator\nContext: Short-form video creation\nFocus: Trends, follower growth\nOutput: 10,000 followers in 6 months",
    ),
    (
        "As an influencer manager, identify 20 micro-influencers for the brand campaign, negotiate collaboration terms, and track influencer performance metrics.",
        "Role: Influencer Manager\nContext: Micro-influencer campaign\nFocus: Negotiation, performance tracking\nOutput: 20 influencer collaborations",
    ),
    (
        "As an affiliate manager, recruit 50 affiliates for the SaaS referral program, provide marketing materials, and pay commissions monthly based on referred revenue.",
        "Role: Affiliate Manager\nContext: SaaS referral program\nFocus: Affiliate recruitment, commissions\nOutput: 50 recruited affiliates",
    ),
    (
        "As a partnership manager, negotiate co-marketing agreements with 5 complementary SaaS products, plan joint webinars, and track partnership-sourced leads.",
        "Role: Partnership Manager\nContext: Co-marketing agreements\nFocus: Joint webinars, lead tracking\nOutput: 5 partnership agreements",
    ),
    (
        "As a channel manager, enable 10 channel partners to sell the software, provide partner training, and track partner-influenced revenue quarterly.",
        "Role: Channel Manager\nContext: Channel partner enablement\nFocus: Partner training, revenue tracking\nOutput: 10 enabled partners",
    ),
    (
        "As a franchise development manager, qualify franchise candidates, present the franchise disclosure document, and sign 3 new franchise agreements this year.",
        "Role: Franchise Development Manager\nContext: Franchise candidate qualification\nFocus: FDD presentation, agreements\nOutput: 3 signed franchise agreements",
    ),
    (
        "As a licensing manager, negotiate licensing agreements for the patented technology, collect royalty payments, and maintain the licensing database.",
        "Role: Licensing Manager\nContext: Patent licensing\nFocus: Negotiation, royalties\nOutput: Licensing agreements and royalties",
    ),
    (
        "As a merchandising manager, plan the product assortment for the retail stores, negotiate with suppliers, and manage inventory allocation across 20 locations.",
        "Role: Merchandising Manager\nContext: Retail product assortment\nFocus: Supplier negotiation, inventory\nOutput: Assortment plan and allocation",
    ),
    (
        "As a store manager, oversee daily operations of the retail location, manage a team of 15 associates, and achieve monthly sales targets.",
        "Role: Store Manager\nContext: Retail store operations\nFocus: Team management, sales targets\nOutput: Monthly sales achievement",
    ),
    (
        "As an assistant manager, support the store manager in daily operations, open and close the store, and handle customer escalations.",
        "Role: Assistant Manager\nContext: Store operations support\nFocus: Opening/closing, escalations\nOutput: Smooth store operations",
    ),
    (
        "As a sales associate, assist customers on the sales floor, process transactions at the point of sale, and maintain visual merchandising standards.",
        "Role: Sales Associate\nContext: Sales floor assistance\nFocus: Transactions, merchandising\nOutput: Processed sales and maintained standards",
    ),
    (
        "As a cashier, process customer payments accurately, balance the cash drawer at end of shift, and provide friendly customer service.",
        "Role: Cashier\nContext: Payment processing\nFocus: Accuracy, cash balancing\nOutput: Balanced drawer and satisfied customers",
    ),
    (
        "As a stock associate, receive merchandise deliveries, stock shelves according to the planogram, and maintain backroom organization.",
        "Role: Stock Associate\nContext: Merchandise stocking\nFocus: Planogram compliance, backroom\nOutput: Stocked shelves and organized backroom",
    ),
    (
        "As a loss prevention officer, monitor CCTV for suspicious activity, apprehend shoplifters per company policy, and write incident reports for management.",
        "Role: Loss Prevention Officer\nContext: CCTV monitoring\nFocus: Shoplifting apprehension, reports\nOutput: Incident reports and loss reduction",
    ),
    (
        "As a visual merchandiser, install window displays for the seasonal collection, arrange floor mannequins, and ensure brand consistency across all stores.",
        "Role: Visual Merchandiser\nContext: Seasonal window displays\nFocus: Brand consistency\nOutput: Installed displays and mannequins",
    ),
    (
        "As a buyer, purchase seasonal merchandise from vendors, negotiate wholesale pricing, and manage the buying budget for the department.",
        "Role: Buyer\nContext: Merchandise purchasing\nFocus: Vendor negotiation, budget\nOutput: Purchased inventory and budget management",
    ),
    (
        "As a planner, forecast sales by SKU for the upcoming season, create purchase orders, and manage inventory levels to minimize stockouts.",
        "Role: Planner\nContext: Seasonal sales forecasting\nFocus: SKU forecasting, POs\nOutput: Purchase orders and inventory plan",
    ),
    (
        "As an allocator, distribute inventory to store locations based on sales performance, execute inter-store transfers, and rebalance stock weekly.",
        "Role: Allocator\nContext: Inventory distribution\nFocus: Inter-store transfers, rebalancing\nOutput: Optimized inventory allocation",
    ),
    (
        "As a supply chain analyst, monitor supplier lead times, identify supply chain risks, and recommend safety stock adjustments to the supply chain manager.",
        "Role: Supply Chain Analyst\nContext: Supplier lead time monitoring\nFocus: Risk identification, safety stock\nOutput: Risk report and stock adjustments",
    ),
    (
        "As a logistics coordinator, schedule inbound and outbound shipments, track deliveries in the TMS, and resolve shipping exceptions with carriers.",
        "Role: Logistics Coordinator\nContext: Shipment scheduling\nFocus: TMS tracking, exception resolution\nOutput: Scheduled and tracked shipments",
    ),
    (
        "As a warehouse manager, oversee daily warehouse operations, manage a team of 20 warehouse associates, and maintain 99.5% inventory accuracy.",
        "Role: Warehouse Manager\nContext: Warehouse operations\nFocus: Team management, accuracy\nOutput: Accurate inventory and efficient ops",
    ),
    (
        "As a forklift operator, move pallets from receiving to storage locations, load outbound trucks, and perform daily forklift safety checks.",
        "Role: Forklift Operator\nContext: Pallet movement\nFocus: Safety checks, loading\nOutput: Moved pallets and loaded trucks",
    ),
    (
        "As a picker, pick customer orders from warehouse shelves using the pick list, scan items for accuracy, and place items in packing stations.",
        "Role: Picker\nContext: Order picking\nFocus: Accuracy, scanning\nOutput: Accurately picked orders",
    ),
    (
        "As a packer, pack customer orders in appropriate boxes, add packing slips and labels, and hand off packages to the shipping carrier.",
        "Role: Packer\nContext: Order packing\nFocus: Packing slips, labeling\nOutput: Packed and labeled orders",
    ),
    (
        "As a shipping clerk, prepare shipping documents, schedule carrier pickups, and update the order management system with tracking numbers.",
        "Role: Shipping Clerk\nContext: Shipping documentation\nFocus: Carrier pickups, tracking\nOutput: Shipped orders with tracking",
    ),
    (
        "As a receiving clerk, unload inbound deliveries, inspect goods for damage, and log received items into the inventory system.",
        "Role: Receiving Clerk\nContext: Inbound delivery inspection\nFocus: Damage inspection, logging\nOutput: Received and logged inventory",
    ),
    (
        "As an inventory control specialist, perform cycle counts for 500 SKUs monthly, investigate variances, and update inventory records in the ERP.",
        "Role: Inventory Control Specialist\nContext: Cycle counts\nFocus: Variance investigation, ERP updates\nOutput: Monthly cycle count report",
    ),
    (
        "As a demand planner, forecast product demand using historical sales data, collaborate with sales on promotions, and update the demand plan monthly.",
        "Role: Demand Planner\nContext: Product demand forecasting\nFocus: Historical data, promo collaboration\nOutput: Monthly demand plan",
    ),
    (
        "As a production planner, create the weekly production schedule for the manufacturing line, balance machine capacity, and expedite rush orders.",
        "Role: Production Planner\nContext: Manufacturing scheduling\nFocus: Capacity balancing, rush orders\nOutput: Weekly production schedule",
    ),
    (
        "As a manufacturing engineer, design the assembly line layout, select equipment, and implement lean manufacturing principles to reduce waste.",
        "Role: Manufacturing Engineer\nContext: Assembly line design\nFocus: Layout, lean principles\nOutput: Optimized assembly line",
    ),
    (
        "As a process engineer, optimize the injection molding process, reduce cycle time by 15%, and document standard operating procedures.",
        "Role: Process Engineer\nContext: Injection molding optimization\nFocus: Cycle time reduction, SOPs\nOutput: 15% faster cycle and SOPs",
    ),
    (
        "As a quality engineer, perform incoming inspection on raw materials, conduct in-process quality checks, and issue non-conformance reports when needed.",
        "Role: Quality Engineer\nContext: Manufacturing quality\nFocus: Inspections, non-conformance\nOutput: Quality reports and NCRs",
    ),
    (
        "As a maintenance planner, schedule preventive maintenance for production equipment, manage spare parts inventory, and minimize unplanned downtime.",
        "Role: Maintenance Planner\nContext: Equipment PM scheduling\nFocus: Spare parts, downtime reduction\nOutput: PM schedule and downtime log",
    ),
    (
        "As a maintenance technician, perform preventive maintenance on CNC machines, troubleshoot electrical faults, and complete work orders in the CMMS.",
        "Role: Maintenance Technician\nContext: CNC machine maintenance\nFocus: Troubleshooting, work orders\nOutput: Completed maintenance tasks",
    ),
    (
        "As an EHS manager, conduct safety training for all employees, perform workplace hazard assessments, and maintain OSHA compliance records.",
        "Role: EHS Manager\nContext: Safety training and compliance\nFocus: Hazard assessments, OSHA records\nOutput: Trained employees and compliance docs",
    ),
    (
        "As a sustainability coordinator, track utility consumption across facilities, implement energy-saving initiatives, and report carbon emissions annually.",
        "Role: Sustainability Coordinator\nContext: Utility tracking\nFocus: Energy initiatives, carbon reporting\nOutput: Annual emissions report",
    ),
    (
        "As a facilities coordinator, book meeting rooms, manage office supply orders, and coordinate building maintenance with the landlord.",
        "Role: Facilities Coordinator\nContext: Office facility management\nFocus: Room booking, supplies\nOutput: Managed facilities and supplies",
    ),
    (
        "As a mailroom clerk, sort and distribute incoming mail, process outgoing shipments, and maintain the mail log for audit purposes.",
        "Role: Mailroom Clerk\nContext: Mail sorting and distribution\nFocus: Outgoing shipments, mail log\nOutput: Sorted mail and audit log",
    ),
    (
        "As a copy center operator, print and bind client presentations, operate large-format plotters, and charge jobs to the correct cost centers.",
        "Role: Copy Center Operator\nContext: Printing and binding\nFocus: Plotter operation, cost center\nOutput: Printed and bound documents",
    ),
    (
        "As a travel coordinator, book flights and hotels for business travelers, process travel expense reports, and enforce the corporate travel policy.",
        "Role: Travel Coordinator\nContext: Business travel booking\nFocus: Expense reports, policy compliance\nOutput: Booked travel and processed expenses",
    ),
    (
        "As an immigration specialist, prepare H-1B visa petitions for foreign employees, track case status with USCIS, and maintain the immigration case database.",
        "Role: Immigration Specialist\nContext: H-1B petition preparation\nFocus: USCIS tracking, case database\nOutput: Filed petitions and case records",
    ),
    (
        "As a global mobility manager, manage employee international assignments, arrange housing and schooling, and handle tax equalization for expatriates.",
        "Role: Global Mobility Manager\nContext: International assignments\nFocus: Housing, schooling, taxes\nOutput: Managed expatriate assignments",
    ),
    (
        "As a payroll specialist, calculate monthly commissions for the sales team, process salary adjustments, and distribute payslips securely.",
        "Role: Payroll Specialist\nContext: Commission calculation\nFocus: Salary adjustments, payslips\nOutput: Processed commissions and payslips",
    ),
    (
        "As a benefits coordinator, process new hire benefits enrollments, manage COBRA administration, and answer employee benefits questions via the help desk.",
        "Role: Benefits Coordinator\nContext: Benefits enrollment\nFocus: COBRA, help desk\nOutput: Processed enrollments and answers",
    ),
    (
        "As a workforce analyst, track absenteeism and turnover rates by department, identify trends, and present workforce metrics to HR leadership monthly.",
        "Role: Workforce Analyst\nContext: Absenteeism and turnover tracking\nFocus: Trend identification\nOutput: Monthly workforce metrics report",
    ),
    (
        "As an HRIS administrator, maintain employee records in the HRIS, generate headcount reports, and support the annual open enrollment in the system.",
        "Role: HRIS Administrator\nContext: HRIS record maintenance\nFocus: Headcount reports, open enrollment\nOutput: Accurate HRIS data and reports",
    ),
    (
        "As a recruitment marketing manager, manage the careers page and job board postings, run employer branding campaigns, and track cost-per-hire by channel.",
        "Role: Recruitment Marketing Manager\nContext: Careers page and job boards\nFocus: Employer branding, cost-per-hire\nOutput: Branded careers page and metrics",
    ),
    (
        "As a candidate experience manager, survey candidates after interviews, analyze feedback, and improve the interview process based on candidate input.",
        "Role: Candidate Experience Manager\nContext: Candidate feedback surveys\nFocus: Process improvement\nOutput: Improved interview process",
    ),
    (
        "As an employer branding specialist, create employee testimonial videos, manage the company Glassdoor profile, and participate in university career fairs.",
        "Role: Employer Branding Specialist\nContext: Employer branding\nFocus: Testimonials, Glassdoor, career fairs\nOutput: Enhanced employer brand presence",
    ),
    (
        "As a diversity recruiter, partner with HBCUs and diversity organizations, attend diversity recruiting events, and increase underrepresented candidate pipeline by 25%.",
        "Role: Diversity Recruiter\nContext: Diversity recruiting partnerships\nFocus: Event attendance, pipeline growth\nOutput: 25% increased diverse pipeline",
    ),
    (
        "As a university relations manager, manage the campus recruiting program, attend 15 career fairs, and convert 20 interns to full-time hires annually.",
        "Role: University Relations Manager\nContext: Campus recruiting\nFocus: Career fairs, intern conversion\nOutput: 20 full-time hires from interns",
    ),
    (
        "As an executive recruiter, conduct retained searches for C-suite positions, assess candidates against leadership competencies, and present a shortlist to the board.",
        "Role: Executive Recruiter\nContext: C-suite retained search\nFocus: Leadership assessment\nOutput: Board-ready candidate shortlist",
    ),
    (
        "As a background check specialist, verify candidate employment history, run criminal background checks, and deliver clearance reports to hiring managers within 5 days.",
        "Role: Background Check Specialist\nContext: Candidate verification\nFocus: Employment history, criminal checks\nOutput: Clearance reports",
    ),
    (
        "As a drug testing coordinator, schedule pre-employment drug tests, coordinate with testing laboratories, and maintain confidentiality of results.",
        "Role: Drug Testing Coordinator\nContext: Pre-employment drug testing\nFocus: Lab coordination, confidentiality\nOutput: Scheduled tests and confidential results",
    ),
    (
        "As an I-9 administrator, complete Section 2 of Form I-9 for new hires, verify identity and employment authorization documents, and maintain I-9 files for audit.",
        "Role: I-9 Administrator\nContext: Form I-9 completion\nFocus: Document verification, audit files\nOutput: Completed I-9 forms",
    ),
    (
        "As a workers' compensation coordinator, manage workers' comp claims, coordinate with injured employees and medical providers, and report claims to the insurance carrier.",
        "Role: Workers' Compensation Coordinator\nContext: Claims management\nFocus: Provider coordination, carrier reporting\nOutput: Managed workers' comp claims",
    ),
    (
        "As a leave of absence manager, process FMLA and personal leave requests, track leave balances, and communicate return-to-work requirements to employees.",
        "Role: LOA Manager\nContext: Leave request processing\nFocus: FMLA, return-to-work\nOutput: Managed leaves and communications",
    ),
    (
        "As a disability management specialist, manage short-term and long-term disability claims, work with employees on return-to-work plans, and coordinate with insurance carriers.",
        "Role: Disability Management Specialist\nContext: Disability claims\nFocus: Return-to-work plans, carrier coordination\nOutput: Managed disability claims",
    ),
    (
        "As an EAP coordinator, promote the Employee Assistance Program, schedule counseling sessions for employees, and track EAP utilization quarterly.",
        "Role: EAP Coordinator\nContext: EAP promotion\nFocus: Session scheduling, utilization\nOutput: Quarterly EAP utilization report",
    ),
    (
        "As a wellness program manager, organize the annual health fair, run the step challenge, and track employee wellness program participation.",
        "Role: Wellness Program Manager\nContext: Health fair and challenges\nFocus: Participation tracking\nOutput: Wellness program participation metrics",
    ),
    (
        "As a recognition program manager, manage the employee recognition platform, approve recognition nominations, and celebrate service milestones quarterly.",
        "Role: Recognition Program Manager\nContext: Recognition platform\nFocus: Nomination approvals, milestones\nOutput: Recognized employees and celebrations",
    ),
    (
        "As a culture ambassador, organize team-building events, promote company values, and gather employee feedback on workplace culture.",
        "Role: Culture Ambassador\nContext: Team building\nFocus: Value promotion, feedback\nOutput: Team events and culture insights",
    ),
    (
        "As a DEI data analyst, collect and analyze diversity hiring data, create DEI dashboards, and present findings to the DEI steering committee.",
        "Role: DEI Data Analyst\nContext: Diversity hiring data\nFocus: Dashboard creation\nOutput: DEI dashboards and committee findings",
    ),
    (
        "As an accessibility specialist, audit the company website for WCAG 2.1 compliance, recommend remediation steps, and work with developers to implement fixes.",
        "Role: Accessibility Specialist\nContext: WCAG 2.1 audit\nFocus: Remediation, developer collaboration\nOutput: Accessibility compliance report",
    ),
    (
        "As a language services manager, manage the translation vendor roster, negotiate translation rates, and ensure multilingual content quality across 12 languages.",
        "Role: Language Services Manager\nContext: Translation vendor management\nFocus: Rate negotiation, quality assurance\nOutput: Managed multilingual content",
    ),
    (
        "As a localization project manager, manage the software localization process for 5 languages, coordinate with translators and engineers, and release localized versions on schedule.",
        "Role: Localization Project Manager\nContext: Software localization\nFocus: Coordination, scheduled releases\nOutput: Localized software versions",
    ),
    (
        "As an international HR manager, manage HR policies for 8 countries, ensure compliance with local labor laws, and support expatriate compensation.",
        "Role: International HR Manager\nContext: Multi-country HR policies\nFocus: Local law compliance\nOutput: Compliant global HR policies",
    ),
    (
        "As a global payroll manager, process payroll for employees in 12 countries, manage local tax filings, and coordinate with local payroll providers.",
        "Role: Global Payroll Manager\nContext: Multi-country payroll\nFocus: Tax filings, provider coordination\nOutput: Processed global payroll",
    ),
    (
        "As an expatriate tax specialist, prepare tax equalization calculations for expatriates, coordinate with local tax advisors, and file host-country tax returns.",
        "Role: Expatriate Tax Specialist\nContext: Expat tax equalization\nFocus: Calculations, local filings\nOutput: Filed expat tax returns",
    ),
    (
        "As a global mobility specialist, arrange visa and work permit sponsorship for international assignees, coordinate with immigration lawyers, and track assignment timelines.",
        "Role: Global Mobility Specialist\nContext: Visa and work permit management\nFocus: Lawyer coordination, timelines\nOutput: Sponsored assignees",
    ),
    (
        "As a relocation specialist, arrange household goods shipping for relocating employees, provide destination services, and track relocation expenses.",
        "Role: Relocation Specialist\nContext: Employee relocation\nFocus: Shipping, destination services\nOutput: Completed relocations",
    ),
    (
        "As a global benefits manager, design benefits programs for employees in 10 countries, negotiate with local insurers, and manage the global benefits budget.",
        "Role: Global Benefits Manager\nContext: Multi-country benefits design\nFocus: Insurer negotiation, budget\nOutput: Global benefits programs",
    ),
    (
        "As an international compensation manager, design expatriate compensation packages, conduct cost-of-living analyses, and manage the global mobility compensation budget.",
        "Role: International Compensation Manager\nContext: Expat compensation\nFocus: Cost-of-living analysis\nOutput: Expat compensation packages",
    ),
    (
        "As a global mobility data analyst, track international assignment costs, analyze assignment success rates, and report global mobility metrics to leadership quarterly.",
        "Role: Global Mobility Data Analyst\nContext: Assignment cost tracking\nFocus: Success rates, metrics\nOutput: Quarterly global mobility report",
    ),
    (
        "As a global HR operations manager, standardize HR processes across 15 countries, implement the global HRIS, and train local HR teams on new processes.",
        "Role: Global HR Operations Manager\nContext: Process standardization\nFocus: HRIS implementation, training\nOutput: Standardized global HR processes",
    ),
    (
        "As a global employee relations manager, handle employee relations cases across multiple countries, ensure compliance with local laws, and maintain consistent case management.",
        "Role: Global Employee Relations Manager\nContext: Multi-country ER cases\nFocus: Local law compliance\nOutput: Managed global ER cases",
    ),
    (
        "As a global learning manager, localize training content for 8 languages, manage the global LMS, and track training completion by region.",
        "Role: Global Learning Manager\nContext: Training localization\nFocus: LMS management, regional tracking\nOutput: Localized training and completion data",
    ),
    (
        "As a global talent manager, identify high-potential employees globally, manage the global succession plan, and present talent reviews to the executive committee.",
        "Role: Global Talent Manager\nContext: High-potential identification\nFocus: Succession planning\nOutput: Global talent review presentations",
    ),
    (
        "As a global mobility tax manager, manage tax equalization for 50+ expatriates, coordinate with global tax advisors, and ensure compliance with host and home country tax laws.",
        "Role: Global Mobility Tax Manager\nContext: Expat tax equalization\nFocus: Multi-country tax compliance\nOutput: Managed expat tax compliance",
    ),
    (
        "As a global immigration counsel, advise on immigration strategy for the global workforce, manage relationships with government authorities, and oversee complex visa cases.",
        "Role: Global Immigration Counsel\nContext: Global immigration strategy\nFocus: Government relations, complex cases\nOutput: Immigration strategy and case management",
    ),
    (
        "As a global payroll analyst, reconcile payroll data across 10 countries, resolve payroll discrepancies, and prepare global payroll journals for finance.",
        "Role: Global Payroll Analyst\nContext: Multi-country payroll reconciliation\nFocus: Discrepancy resolution\nOutput: Reconciled payroll journals",
    ),
    (
        "As a global benefits analyst, analyze benefits utilization by country, recommend plan design changes, and manage the global benefits open enrollment.",
        "Role: Global Benefits Analyst\nContext: Benefits utilization analysis\nFocus: Plan design, open enrollment\nOutput: Utilization report and plan changes",
    ),
    (
        "As a global mobility operations manager, manage the end-to-end relocation process for 100+ assignees annually, oversee relocation vendors, and maintain the mobility budget.",
        "Role: Global Mobility Operations Manager\nContext: Relocation process management\nFocus: Vendor oversight, budget\nOutput: Managed relocations and budget",
    ),
    (
        "As a global workforce planning manager, forecast global headcount by region and function, align workforce plans with business strategy, and present to the board annually.",
        "Role: Global Workforce Planning Manager\nContext: Global headcount forecasting\nFocus: Strategic alignment\nOutput: Annual board workforce presentation",
    ),
    (
        "As a global HR business partner, support the APAC leadership team on people matters, coach leaders on performance management, and drive regional HR initiatives.",
        "Role: Global HRBP\nContext: APAC leadership support\nFocus: Performance coaching, initiatives\nOutput: Supported APAC leadership",
    ),
    (
        "As a global HR project manager, lead the implementation of the new global performance management system, coordinate with regional stakeholders, and manage the project timeline.",
        "Role: Global HR Project Manager\nContext: Performance system implementation\nFocus: Stakeholder coordination\nOutput: Implemented performance system",
    ),
    (
        "As a global HR change manager, lead change management for the global HR transformation, develop communication plans, and assess readiness across regions.",
        "Role: Global HR Change Manager\nContext: HR transformation\nFocus: Communication, readiness\nOutput: Change management plan and readiness assessment",
    ),
    (
        "As a global HR data analyst, consolidate people data from 12 countries, create global workforce dashboards, and provide data-driven insights to the CHRO.",
        "Role: Global HR Data Analyst\nContext: Global people data consolidation\nFocus: Dashboards, insights\nOutput: Global workforce dashboards",
    ),
    (
        "As a global HR technology manager, manage the global HR tech stack, evaluate new HR software vendors, and oversee system integrations.",
        "Role: Global HR Technology Manager\nContext: HR tech stack management\nFocus: Vendor evaluation, integrations\nOutput: Optimized global HR technology",
    ),
    (
        "As a global HR vendor manager, manage contracts with global HR vendors, negotiate service level agreements, and conduct annual vendor performance reviews.",
        "Role: Global HR Vendor Manager\nContext: Vendor contract management\nFocus: SLA negotiation, reviews\nOutput: Managed vendor relationships",
    ),
    (
        "As a global HR compliance manager, ensure HR compliance in 20 countries, monitor regulatory changes, and update global HR policies annually.",
        "Role: Global HR Compliance Manager\nContext: Multi-country compliance\nFocus: Regulatory monitoring\nOutput: Compliant global HR policies",
    ),
    (
        "As a global HR audit manager, conduct HR audits in 5 countries per year, assess compliance with local laws, and remediate findings with local HR teams.",
        "Role: Global HR Audit Manager\nContext: Multi-country HR audits\nFocus: Compliance assessment\nOutput: Audit reports and remediation",
    ),
    (
        "As a global HR reporting manager, prepare monthly HR metrics for the executive team, track headcount and attrition globally, and present trends quarterly.",
        "Role: Global HR Reporting Manager\nContext: Global HR metrics\nFocus: Headcount, attrition\nOutput: Quarterly trends presentation",
    ),
    (
        "As a global HR budget manager, manage the annual HR budget for 15 countries, track spending by category, and provide monthly budget variance reports.",
        "Role: Global HR Budget Manager\nContext: Multi-country budget management\nFocus: Spending tracking\nOutput: Monthly variance reports",
    ),
    (
        "As a global HR procurement manager, procure HR software and services globally, negotiate enterprise agreements, and manage vendor relationships.",
        "Role: Global HR Procurement Manager\nContext: Global HR procurement\nFocus: Enterprise agreements\nOutput: Procured HR solutions",
    ),
    (
        "As a global HR legal counsel, advise on global employment law issues, review employment contracts, and manage employment litigation risk.",
        "Role: Global HR Legal Counsel\nContext: Global employment law\nFocus: Contract review, risk management\nOutput: Legal advice and risk mitigation",
    ),
    (
        "As a global HR risk manager, identify HR-related risks across the organization, develop risk mitigation strategies, and report to the risk committee quarterly.",
        "Role: Global HR Risk Manager\nContext: HR risk identification\nFocus: Mitigation strategies\nOutput: Quarterly risk reports",
    ),
    (
        "As a global HR internal communications manager, communicate global HR initiatives to employees in 10 languages, manage the global intranet, and ensure message consistency.",
        "Role: Global HR Internal Communications Manager\nContext: Multilingual communications\nFocus: Intranet, consistency\nOutput: Consistent global communications",
    ),
    (
        "As a global HR events manager, plan the annual global leadership summit, coordinate travel for 200 executives, and manage the event budget.",
        "Role: Global HR Events Manager\nContext: Leadership summit planning\nFocus: Travel coordination, budget\nOutput: Executed global summit",
    ),
    (
        "As a global HR recognition manager, manage the global employee recognition program, translate recognition materials into 8 languages, and track global recognition activity.",
        "Role: Global HR Recognition Manager\nContext: Global recognition program\nFocus: Translation, activity tracking\nOutput: Managed global recognition",
    ),
    (
        "As a global HR wellness manager, design wellness programs for a global workforce, adapt programs for cultural differences, and track global participation.",
        "Role: Global HR Wellness Manager\nContext: Global wellness design\nFocus: Cultural adaptation\nOutput: Global wellness programs",
    ),
    (
        "As a global HR learning manager, design leadership development programs for global high-potentials, deliver programs in multiple languages, and measure leadership readiness.",
        "Role: Global HR Learning Manager\nContext: Global leadership development\nFocus: Multilingual delivery\nOutput: Developed global leaders",
    ),
    (
        "As a global HR succession manager, manage the global succession planning process, identify ready-now successors for critical roles, and present succession plans to the board.",
        "Role: Global HR Succession Manager\nContext: Global succession planning\nFocus: Successor identification\nOutput: Board succession presentations",
    ),
    (
        "As a global HR career development manager, create global career pathing frameworks, deliver career coaching sessions, and track internal mobility rates.",
        "Role: Global HR Career Development Manager\nContext: Career pathing frameworks\nFocus: Internal mobility\nOutput: Career frameworks and mobility data",
    ),
    (
        "As a global HR performance manager, design the global performance review process, calibrate ratings across regions, and manage the annual review cycle.",
        "Role: Global HR Performance Manager\nContext: Global performance reviews\nFocus: Calibration, annual cycle\nOutput: Calibrated performance process",
    ),
    (
        "As a global HR compensation manager, design global compensation structures, manage salary bands by country, and oversee the global merit increase process.",
        "Role: Global HR Compensation Manager\nContext: Global comp structures\nFocus: Salary bands, merit process\nOutput: Global compensation framework",
    ),
    (
        "As a global HR payroll manager, oversee global payroll operations, ensure compliance with local payroll laws, and manage the global payroll vendor.",
        "Role: Global HR Payroll Manager\nContext: Global payroll operations\nFocus: Law compliance, vendor mgmt\nOutput: Compliant global payroll",
    ),
    (
        "As a global HR time and attendance manager, implement global time tracking policies, manage absence policies by country, and oversee the timekeeping system.",
        "Role: Global HR Time and Attendance Manager\nContext: Time tracking policies\nFocus: Absence policies\nOutput: Managed global timekeeping",
    ),
    (
        "As a global HR workforce analytics manager, build predictive attrition models by region, analyze workforce productivity, and provide predictive insights to CHRO.",
        "Role: Global HR Workforce Analytics Manager\nContext: Predictive attrition modeling\nFocus: Regional analysis\nOutput: Predictive workforce insights",
    ),
]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_dataset() -> dict:
    """Assemble the expanded mock dataset into a structured dictionary."""
    vague_pairs = [{"input": p, "target": q} for p, q in VAGUE_TO_QUESTIONS]
    clear_pairs = [{"input": p, "target": f} for p, f in CLEAR_TO_RCFO]

    return {
        "version": "3.0.0",
        "description": (
            "Expanded mock dataset for Inspector AI critical thinking training. "
            "Part 1 maps vague prompts to clarification questions; "
            "Part 2 maps clear prompts to R.C.F.O formatted prompts."
        ),
        "splits": {
            "vague_to_questions": vague_pairs,
            "clear_to_rcfo": clear_pairs,
        },
        "counts": {
            "vague_to_questions": len(vague_pairs),
            "clear_to_rcfo": len(clear_pairs),
            "total": len(vague_pairs) + len(clear_pairs),
        },
    }


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset()

    save_json(dataset, RAW_DATA_DIR / "mock_dataset.json")
    save_json(dataset, PROCESSED_DATA_DIR / "mock_dataset.json")

    print("Inspector AI — expanded mock dataset built successfully.")
    print(f"Vague -> Questions pairs: {dataset['counts']['vague_to_questions']}")
    print(f"Clear -> R.C.F.O pairs:   {dataset['counts']['clear_to_rcfo']}")
    print(f"Total pairs:              {dataset['counts']['total']}")
    print(f"Written to: {RAW_DATA_DIR / 'mock_dataset.json'}")
    print(f"Written to: {PROCESSED_DATA_DIR / 'mock_dataset.json'}")


if __name__ == "__main__":
    main()
