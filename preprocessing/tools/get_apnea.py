# path = 'E:\\Capstone\\shhs\\polysomnography\\annotations-events-nsrr\\shhs1\\shhs1-200001-nsrr.xml'

def get_apnea(path):
    from lxml import etree
    tree = etree.parse(path)
    root = tree.getroot()
    events = root.xpath("//ScoredEvent[EventConcept[contains(text(), 'pnea')]]/Start | //ScoredEvent[EventConcept[contains(text(), 'pnea')]]/Duration")

    # cnt = 1
    result = []
    start = 0.0
    for event in events:
        # print("{:>8} : {:<8}".format(event.tag, event.text))
        if event.tag == 'Start':
            start = float(event.text)
        elif event.tag == 'Duration':
            duration = float(event.text)
            end = start + duration
            start, end, duration = map(round, [start, end, duration])
            # print('{:>3}st : {:^7.1f} to {:^7.1f}'.format(cnt, start, end))
            # cnt += 1
            result.append((start, end, duration))
    return result