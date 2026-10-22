"""The shipped examples build and serve, so a change to the package cannot break them silently."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The examples open their SQLite file in the working directory."""
    monkeypatch.chdir(tmp_path)


def test_quickstart_continues_a_conversation() -> None:
    from pydantic_ai_conversations_examples import quickstart

    with TestClient(quickstart.app) as client:
        first = client.post('/run', json={'prompt': 'My printer is offline.', 'conversation_id': 'c1'})
        assert first.status_code == 200
        second = client.post(
            '/run',
            json={
                'prompt': 'It is a Brother.',
                'conversation_id': 'c1',
                'extras': {'previous_run_id': first.json()['run_id']},
            },
        )
        assert second.status_code == 200
        turns = client.get('/conversations/c1').json()['turns']
        assert [t['previous_run_id'] for t in turns] == [None, first.json()['run_id']]


def test_shared_cases_lets_members_share_and_keeps_others_out() -> None:
    from pydantic_ai_conversations_examples import shared_cases

    with TestClient(shared_cases.app) as client:
        created = client.post(
            '/conversations',
            params={'case_id': 'case-9'},
            json={'attrs': {'topic': 'billing'}},
            headers={'X-User': 'alice'},
        )
        assert created.status_code == 201
        conversation_id = created.json()['conversation_id']

        chat = {
            'id': conversation_id,
            'trigger': 'submit-message',
            'caseId': 'case-9',
            'messages': [{'id': 'm1', 'role': 'user', 'parts': [{'type': 'text', 'text': 'Double charge.'}]}],
        }
        assert client.post('/chat', json=chat, headers={'X-User': 'alice'}).status_code == 200
        listed = client.get(
            '/conversations', params={'case_id': 'case-9', 'topic': 'billing'}, headers={'X-User': 'bob'}
        )
        [entry] = listed.json()['data']
        assert entry['conversation_id'] == conversation_id

        run_id = client.get(
            f'/conversations/{conversation_id}', params={'case_id': 'case-9'}, headers={'X-User': 'bob'}
        ).json()['turns'][0]['run_id']
        followup = {
            'prompt': 'Refund issued.',
            'conversation_id': conversation_id,
            'extras': {'case_id': 'case-9', 'previous_run_id': run_id},
        }
        assert client.post('/run', json=followup, headers={'X-User': 'bob'}).status_code == 200

        detail = client.get(f'/conversations/{conversation_id}', params={'case_id': 'case-9'}).json()
        assert [t['author_user_id'] for t in detail['turns']] == ['alice', 'bob']
        assert all(t['ui_messages'] for t in detail['turns'])

        assert (
            client.get('/conversations', params={'case_id': 'case-9'}, headers={'X-User': 'carol'}).status_code == 403
        )
        assert client.post('/run', json=followup, headers={'X-User': 'carol'}).status_code == 403
        assert client.get('/conversations', headers={'X-User': 'mallory'}).status_code == 401
        # A member's case in the query does not cover a different case in the body.
        other_case = {**chat, 'caseId': 'case-7'}
        assert client.post('/chat', params={'case_id': 'case-9'}, json=other_case).status_code == 403

        deleted = client.delete(f'/conversations/{conversation_id}', params={'case_id': 'case-9'})
        assert deleted.status_code == 204
        assert shared_cases.forgotten == [conversation_id]


def test_ticket_attrs_are_set_by_the_server_and_filterable() -> None:
    from pydantic_ai_conversations_examples import ticket_attrs

    with TestClient(ticket_attrs.app) as client:
        created = client.post('/tickets/T-17/conversations')
        assert created.status_code == 201
        conversation_id = created.json()['conversation_id']
        assert created.json()['attrs'] == {'ticket': 'T-17', 'product': 'printer', 'priority': 'high'}
        client.post('/tickets/T-18/conversations')

        run = client.post('/run', json={'prompt': 'The printer is offline.', 'conversation_id': conversation_id})
        assert run.status_code == 200

        [entry] = client.get('/conversations', params={'priority': 'high'}).json()['data']
        assert entry['conversation_id'] == conversation_id
        assert entry['attrs']['ticket'] == 'T-17'
        assert client.post('/tickets/T-99/conversations').status_code == 404
        # Without the create route a client cannot choose attrs.
        assert client.post('/conversations', json={'attrs': {'priority': 'high'}}).status_code == 405
