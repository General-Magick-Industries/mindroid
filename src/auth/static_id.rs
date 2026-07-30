use async_trait::async_trait;

use crate::auth::CredentialKind;
use crate::{Auth, Result};

pub struct StaticAuth {
    token: String,
    kind: CredentialKind,
}

impl StaticAuth {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            kind: CredentialKind::ServiceUser,
        }
    }

    /// A static credential that authenticates as an end user (a bifrost
    /// end-user JWT): it targets the `/v1/end-user/...` routes and rides in the
    /// Centrifugo connect frame's `data` field, routed to the connect proxy.
    pub fn new_end_user(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            kind: CredentialKind::EndUser,
        }
    }
}

#[async_trait]
impl Auth for StaticAuth {
    async fn get_token(&self) -> Result<String> {
        Ok(self.token.clone())
    }

    async fn get_auth_headers(&self) -> Result<Vec<(String, String)>> {
        Ok(vec![(
            "Authorization".to_string(),
            format!("Bearer {}", self.token),
        )])
    }

    fn is_authenticated(&self) -> bool {
        true
    }

    async fn refresh(&self) -> Result<()> {
        Ok(())
    }

    fn credential_kind(&self) -> CredentialKind {
        self.kind
    }
}
