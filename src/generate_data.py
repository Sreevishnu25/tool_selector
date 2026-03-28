import json
import random
import os

# Expanded templates for each tool
templates = {
    'compose_new_email': [
        'Send an email to {person} about {topic}',
        'Write an email to {person} regarding {topic}',
        'Email {person} about {topic}',
        'Draft an email to {person} about {topic}',
        'Compose an email to {person} regarding {topic}',
        'Send a message to {person} via email about {topic}',
        'I need to email {person} about {topic}',
        'Can you send an email to {person} about {topic}',
        'Please write an email to {person} regarding {topic}',
        'Shoot an email to {person} about {topic}',
        'Drop an email to {person} about {topic}',
        'Send {person} an email about {topic}',
        'Mail {person} about {topic}',
        'Write to {person} about {topic}',
        'Contact {person} by email about {topic}',
    ],
    'reply_to_email': [
        'Reply to {person} about {topic}',
        'Respond to {person} email about {topic}',
        'Answer {person} email regarding {topic}',
        'Send a reply to {person} about {topic}',
        'Reply back to {person} regarding {topic}',
        'Write a response to {person} about {topic}',
        'Get back to {person} about {topic}',
        'Respond back to {person} email on {topic}',
        'Send a reply email to {person} about {topic}',
        'Reply to the email from {person} about {topic}',
        'Write back to {person} regarding {topic}',
        'Answer back to {person} about {topic}',
    ],
    'forward_email': [
        'Forward the email to {person}',
        'Forward {topic} email to {person}',
        'Send the {topic} email to {person}',
        'Forward the message about {topic} to {person}',
        'Pass along the {topic} email to {person}',
        'Forward this email about {topic} to {person}',
        'Send the email regarding {topic} to {person}',
        'Forward {person} the email about {topic}',
        'Redirect the {topic} email to {person}',
        'Share the {topic} email with {person}',
    ],
    'send_sms': [
        'Text {person} that {message}',
        'Send a text to {person} saying {message}',
        'Message {person} that {message}',
        'Send SMS to {person} about {topic}',
        'Text message {person} that {message}',
        'Send a text message to {person} saying {message}',
        'SMS {person} that {message}',
        'Shoot a text to {person} saying {message}',
        'Text {person} about {topic}',
        'Send {person} a text saying {message}',
        'Drop {person} a text about {topic}',
        'Message {person} about {topic}',
        'Send a quick text to {person} about {topic}',
        'Text {person} letting them know {message}',
    ],
    'create_calendar_event': [
        'Schedule a meeting {time} about {topic}',
        'Create a calendar event for {time}',
        'Book a meeting {time} with {person}',
        'Set up a meeting {time} about {topic}',
        'Add {topic} to my calendar {time}',
        'Schedule {topic} for {time}',
        'Put a meeting on my calendar for {time}',
        'Create an event for {topic} {time}',
        'Add {time} meeting to my calendar',
        'Block {time} on my calendar for {topic}',
        'Set up a {topic} meeting for {time}',
        'Calendar event for {topic} {time}',
        'Book time for {topic} {time}',
        'Reserve {time} for {topic} meeting',
        'Plan a meeting about {topic} for {time}',
    ],
    'get_zoom_meeting_link': [
        'Set up a zoom call with {person}',
        'Create a video meeting with {person}',
        'Get a zoom link for the meeting with {person}',
        'Schedule a video call about {topic}',
        'Arrange a zoom meeting with {person}',
        'Get a video conferencing link for {person}',
        'Create a zoom meeting for {topic}',
        'Set up a virtual meeting with {person}',
        'Generate a zoom link for the {topic} meeting',
        'Schedule an online meeting with {person} about {topic}',
        'Create a video call link for {person}',
        'Get zoom details for meeting with {person}',
    ],
    'create_note': [
        'Create a note called {topic}',
        'Make a new note about {topic}',
        'Start a note for {topic}',
        'Create a new note named {topic}',
        'Write a note about {topic}',
        'Take a note about {topic}',
        'Jot down notes about {topic}',
        'Create a note for {topic}',
        'Make a note titled {topic}',
        'Start a new note for {topic}',
        'Write down {topic} in a note',
        'Create notes for {topic}',
        'Begin a note about {topic}',
        'Save a note about {topic}',
    ],
    'open_note': [
        'Open my {topic} note',
        'Show me the {topic} notes',
        'Open the note about {topic}',
        'Find my {topic} note',
        'Pull up my {topic} note',
        'Display my {topic} notes',
        'Get my {topic} note',
        'Read my {topic} note',
        'Access my {topic} notes',
        'View my note about {topic}',
        'Look at my {topic} notes',
        'Retrieve my {topic} note',
    ],
    'append_note_content': [
        'Add {message} to my {topic} note',
        'Append {message} to the {topic} notes',
        'Update my {topic} note with {message}',
        'Add more content to {topic} note',
        'Insert {message} into my {topic} note',
        'Write {message} in my {topic} note',
        'Add to my {topic} note',
        'Update the {topic} notes',
        'Add information to my {topic} note',
        'Append to my {topic} notes',
        'Add new content to {topic} note',
    ],
    'create_reminder': [
        'Remind me to {action} {time}',
        'Set a reminder to {action} {time}',
        'Create a reminder for {action} {time}',
        'Do not let me forget to {action} {time}',
        'Remind me about {topic} {time}',
        'Alert me to {action} {time}',
        'Set an alarm to {action} {time}',
        'Give me a reminder to {action} {time}',
        'Notify me to {action} {time}',
        'Schedule a reminder for {action} {time}',
        'Ping me about {topic} {time}',
        'Remind me {time} to {action}',
        'Set a notification to {action} {time}',
        'Remember to {action} {time}',
    ],
    'summarize_pdf': [
        'Summarize the {topic} PDF',
        'Give me a summary of the {topic} document',
        'Read and summarize {topic} PDF',
        'What does the {topic} PDF say',
        'Summarize the {topic} file',
        'Get the key points from the {topic} PDF',
        'Brief me on the {topic} document',
        'What is in the {topic} PDF',
        'Extract main points from {topic} document',
        'Give summary of {topic} PDF',
        'Digest the {topic} document for me',
        'Outline the {topic} PDF content',
        'Review the {topic} document and summarize',
    ],
    'open_and_get_file_path': [
        'Open the {topic} file',
        'Find the {topic} document',
        'Get the {topic} file',
        'Locate the {topic} document',
        'Access the {topic} file',
        'Retrieve the {topic} document',
        'Pull up the {topic} file',
        'Get the path to the {topic} file',
        'Find where the {topic} file is',
        'Look for the {topic} document',
        'Get my {topic} file',
        'Open the {topic} document for me',
    ],
    'maps_open_location': [
        'Where is {place}',
        'Find {place} on the map',
        'Show me {place}',
        'Locate {place}',
        'Find the location of {place}',
        'Where can I find {place}',
        'Show me where {place} is',
        'What is the location of {place}',
        'Pin {place} on the map',
        'Search for {place} on maps',
        'Look up {place} on the map',
        'Find {place} location',
        'Show location of {place}',
    ],
    'maps_show_direction': [
        'Get directions to {place}',
        'Navigate to {place}',
        'How do I get to {place}',
        'Show me the way to {place}',
        'Directions to {place}',
        'Take me to {place}',
        'Route to {place}',
        'How to reach {place}',
        'Guide me to {place}',
        'Find route to {place}',
        'Best way to get to {place}',
        'Drive me to {place}',
        'Walk me to {place}',
        'Show directions to {place}',
        'Navigate me to {place}',
    ],
    'get_email_address': [],
    'get_phone_number': [],
}

# Expanded fill-in values
persons = [
    'John', 'Sarah', 'Mom', 'Dad', 'Mike', 'Lisa', 'David', 'Emma',
    'Boss', 'Manager', 'HR', 'Client', 'Team', 'Marketing', 'James',
    'Anna', 'Robert', 'Jennifer', 'William', 'Jessica', 'Alex', 'Chris',
    'Priya', 'Raj', 'Arun', 'Divya', 'Meera', 'Kumar', 'Nina', 'Tom',
    'Dr. Smith', 'Professor Lee', 'the supplier', 'the vendor', 'my colleague',
    'the intern', 'the accountant', 'the lawyer', 'the designer', 'the developer'
]

topics = [
    'meeting', 'project', 'report', 'budget', 'deadline', 'presentation',
    'contract', 'proposal', 'schedule', 'update', 'feedback', 'review',
    'plan', 'strategy', 'ideas', 'notes', 'summary', 'invoice', 'sales',
    'quarterly results', 'annual review', 'product launch', 'marketing campaign',
    'technical issue', 'customer complaint', 'performance review', 'leave request',
    'salary discussion', 'team outing', 'onboarding', 'training session',
    'research findings', 'design mockup', 'code review', 'security audit',
    'business trip', 'conference details', 'workshop agenda', 'sprint planning',
]

times = [
    'tomorrow', 'next week', 'on Friday', 'at 3pm', 'tomorrow morning',
    'next Monday', 'this afternoon', 'at 10am', 'in an hour', 'tonight',
    'tomorrow at 2pm', 'next Tuesday', 'on Wednesday', 'at noon',
    'this evening', 'next month', 'on Thursday', 'at 9am', 'at 4pm',
    'next Friday', 'this weekend', 'in 30 minutes', 'at 11am', 'tomorrow evening',
]

messages = [
    'I will be late', 'the meeting is cancelled', 'I am on my way',
    'call me back', 'happy birthday', 'thank you', 'see you soon',
    'I got the package', 'dinner is ready', 'I miss you',
    'the flight is delayed', 'I arrived safely', 'please wait for me',
    'the order is confirmed', 'I need help', 'running 10 minutes late',
    'the project is done', 'I will call later', 'good luck today',
]

actions = [
    'call the dentist', 'submit the report', 'buy groceries',
    'pick up laundry', 'pay bills', 'book flight', 'renew subscription',
    'call mom', 'send invoice', 'review document', 'attend meeting',
    'complete assignment', 'follow up with client', 'check emails',
    'prepare slides', 'update the spreadsheet', 'back up files',
    'take medicine', 'exercise', 'water the plants', 'read the report',
    'sign the contract', 'approve the request', 'finish the task',
]

places = [
    'Central Park', 'the airport', 'nearest coffee shop', 'Times Square',
    'the office', 'downtown', 'the hospital', 'nearest gas station',
    'the mall', 'home', 'the restaurant', 'the gym', 'the library',
    'the bank', 'the school', 'the market', 'the pharmacy', 'the park',
    'the hotel', 'the station', 'the conference center', 'the clinic',
    'the supermarket', 'the bookstore', 'the post office',
]

# Tool combinations
tool_combos = [
    (['compose_new_email'], ['compose_new_email', 'get_email_address']),
    (['reply_to_email'], ['reply_to_email', 'get_email_address']),
    (['forward_email'], ['forward_email', 'get_email_address']),
    (['send_sms'], ['send_sms', 'get_phone_number']),
    (['create_calendar_event'], ['create_calendar_event']),
    (['create_calendar_event'], ['create_calendar_event', 'get_email_address']),
    (['create_calendar_event', 'get_zoom_meeting_link'], ['create_calendar_event', 'get_zoom_meeting_link', 'get_email_address']),
    (['get_zoom_meeting_link'], ['get_zoom_meeting_link', 'get_email_address']),
    (['create_note'], ['create_note']),
    (['open_note'], ['open_note']),
    (['append_note_content'], ['append_note_content', 'open_note']),
    (['create_reminder'], ['create_reminder']),
    (['summarize_pdf'], ['summarize_pdf', 'open_and_get_file_path']),
    (['open_and_get_file_path'], ['open_and_get_file_path']),
    (['maps_open_location'], ['maps_open_location']),
    (['maps_show_direction'], ['maps_show_direction']),
]

random.seed(42)
samples = []

for _ in range(3000):
    template_keys, tools = random.choice(tool_combos)
    main_tool = template_keys[0]
    if not templates[main_tool]:
        continue
    template = random.choice(templates[main_tool])
    try:
        query = template.format(
            person=random.choice(persons),
            topic=random.choice(topics),
            time=random.choice(times),
            message=random.choice(messages),
            action=random.choice(actions),
            place=random.choice(places),
        )
    except KeyError:
        continue
    samples.append({'query': query, 'tools': tools})

random.shuffle(samples)

# 80/10/10 split
train = samples[:2400]
val   = samples[2400:2700]
test  = samples[2700:]

os.makedirs('data', exist_ok=True)
with open('data/train.json', 'w') as f: json.dump(train, f, indent=2)
with open('data/val.json',   'w') as f: json.dump(val,   f, indent=2)
with open('data/test.json',  'w') as f: json.dump(test,  f, indent=2)

print('✅ Dataset created!')
print(f'   Train : {len(train)} samples')
print(f'   Val   : {len(val)} samples')
print(f'   Test  : {len(test)} samples')
print(f'   Total : {len(samples)} samples')

# Tool distribution
from collections import Counter
all_tools = [t for s in samples for t in s['tools']]
print('\n📊 Tool Distribution:')
for tool, count in sorted(Counter(all_tools).items(), key=lambda x: -x[1]):
    print(f'   {tool:<35} {count}')
